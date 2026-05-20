"""
Admin analytics and platform metrics endpoints.

Requires admin role. Provides usage stats, user counts, diagnostic
distribution, and agent performance metrics.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from app.api.deps import AdminUser, DBSession
from app.core.logging import get_logger
from app.models.chat import ChatSession, Message
from app.models.diagnosis import DiagnosisRecord, TriageLevel
from app.models.user import User, UserRole
from app.services.connection_manager import get_connection_manager

router = APIRouter(prefix="/admin", tags=["Admin"])
logger = get_logger(__name__)


# ── Platform Overview ───────────────────────────────────────────────

@router.get(
    "/stats",
    summary="Platform usage statistics",
    response_model=dict[str, Any],
)
async def get_platform_stats(
    admin: AdminUser,
    db: DBSession,
) -> dict[str, Any]:
    """Return comprehensive platform usage statistics.

    Includes user counts, session counts, diagnosis distribution,
    and active WebSocket connections.
    """
    # User counts
    total_users = await db.scalar(select(func.count(User.id)))
    active_users = await db.scalar(
        select(func.count(User.id)).where(User.is_active == True),  # noqa: E712
    )

    # Session counts
    total_sessions = await db.scalar(select(func.count(ChatSession.id)))
    total_messages = await db.scalar(select(func.count(Message.id)))

    # Diagnosis counts
    total_diagnoses = await db.scalar(select(func.count(DiagnosisRecord.id)))

    # Triage distribution
    triage_dist: dict[str, int] = {}
    for level in TriageLevel:
        count = await db.scalar(
            select(func.count(DiagnosisRecord.id)).where(
                DiagnosisRecord.triage_level == level,
            ),
        )
        triage_dist[level.value] = count or 0

    # Average feedback rating
    avg_rating = await db.scalar(
        select(func.avg(DiagnosisRecord.feedback_rating)).where(
            DiagnosisRecord.feedback_rating.isnot(None),
        ),
    )

    # Recent activity (last 24h)
    day_ago = datetime.now(timezone.utc) - timedelta(days=1)
    new_users_24h = await db.scalar(
        select(func.count(User.id)).where(User.created_at >= day_ago),
    )
    new_sessions_24h = await db.scalar(
        select(func.count(ChatSession.id)).where(ChatSession.created_at >= day_ago),
    )
    new_diagnoses_24h = await db.scalar(
        select(func.count(DiagnosisRecord.id)).where(
            DiagnosisRecord.created_at >= day_ago,
        ),
    )

    # WebSocket connections
    ws_manager = get_connection_manager()

    return {
        "users": {
            "total": total_users or 0,
            "active": active_users or 0,
            "new_24h": new_users_24h or 0,
        },
        "sessions": {
            "total": total_sessions or 0,
            "new_24h": new_sessions_24h or 0,
        },
        "messages": {
            "total": total_messages or 0,
        },
        "diagnoses": {
            "total": total_diagnoses or 0,
            "new_24h": new_diagnoses_24h or 0,
            "triage_distribution": triage_dist,
            "average_feedback_rating": round(avg_rating, 2) if avg_rating else None,
        },
        "realtime": {
            "active_ws_connections": ws_manager.active_connections,
            "active_ws_sessions": ws_manager.active_sessions,
        },
    }


# ── User Management ────────────────────────────────────────────────

@router.get(
    "/users",
    summary="List all users (paginated)",
    response_model=dict[str, Any],
)
async def list_users(
    admin: AdminUser,
    db: DBSession,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    role: str | None = Query(default=None, pattern="^(patient|admin)$"),
) -> dict[str, Any]:
    """List all users with optional role filter."""
    query = select(User).order_by(User.created_at.desc()).limit(limit).offset(offset)
    if role:
        query = query.where(User.role == UserRole(role))

    result = await db.execute(query)
    users = result.scalars().all()

    total = await db.scalar(select(func.count(User.id)))

    return {
        "users": [
            {
                "id": str(u.id),
                "email": u.email,
                "full_name": u.full_name,
                "role": u.role.value,
                "auth_provider": u.auth_provider.value,
                "is_active": u.is_active,
                "is_verified": u.is_verified,
                "created_at": u.created_at.isoformat(),
                "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
            }
            for u in users
        ],
        "total": total or 0,
        "limit": limit,
        "offset": offset,
    }


@router.patch(
    "/users/{user_id}/role",
    summary="Change a user's role",
)
async def change_user_role(
    user_id: str,
    role: str = Query(..., pattern="^(patient|admin)$"),
    admin: AdminUser = ...,
    db: DBSession = ...,
) -> dict[str, str]:
    """Promote or demote a user."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = UserRole(role)
    await db.commit()
    logger.info("Admin %s changed user %s role to %s", admin.id, user_id, role)
    return {"detail": f"User {user_id} role changed to {role}"}


@router.patch(
    "/users/{user_id}/deactivate",
    summary="Deactivate a user account",
)
async def deactivate_user(
    user_id: str,
    admin: AdminUser = ...,
    db: DBSession = ...,
) -> dict[str, str]:
    """Soft-delete a user by deactivating their account."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = False
    await db.commit()
    logger.info("Admin %s deactivated user %s", admin.id, user_id)
    return {"detail": f"User {user_id} deactivated"}


# ── Diagnosis Analytics ────────────────────────────────────────────

@router.get(
    "/diagnoses/top-conditions",
    summary="Top diagnosed conditions",
    response_model=list[dict[str, Any]],
)
async def top_conditions(
    admin: AdminUser,
    db: DBSession,
    limit: int = Query(default=20, ge=1, le=50),
) -> list[dict[str, Any]]:
    """Return the most frequently diagnosed conditions."""
    result = await db.execute(
        select(
            DiagnosisRecord.primary_condition,
            func.count(DiagnosisRecord.id).label("count"),
            func.avg(DiagnosisRecord.primary_confidence).label("avg_confidence"),
        )
        .group_by(DiagnosisRecord.primary_condition)
        .order_by(func.count(DiagnosisRecord.id).desc())
        .limit(limit),
    )
    rows = result.all()

    return [
        {
            "condition": row.primary_condition,
            "count": row.count,
            "avg_confidence": round(row.avg_confidence, 3) if row.avg_confidence else None,
        }
        for row in rows
    ]
