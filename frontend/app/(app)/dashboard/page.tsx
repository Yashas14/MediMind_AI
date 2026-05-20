"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/providers/auth-provider";
import {
  chatApi,
  diagnosisApi,
  hospitalApi,
  drugApi,
  type DiagnosisRecord,
  type ChatSession,
} from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  LineChart,
  Line,
} from "recharts";

const TRIAGE_COLORS: Record<string, string> = {
  EMERGENCY: "#ef4444",
  URGENT: "#f97316",
  ROUTINE: "#16a34a",
  SELF_CARE: "#3b82f6",
};

export default function DashboardPage() {
  const router = useRouter();
  const { token, isAuthenticated, isLoading } = useAuth();
  const [diagnoses, setDiagnoses] = useState<DiagnosisRecord[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [healthSummary, setHealthSummary] = useState<any>(null);
  const [drugSearch, setDrugSearch] = useState("");
  const [drugResult, setDrugResult] = useState<any>(null);
  const [drugLoading, setDrugLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"overview" | "history" | "drugs">(
    "overview"
  );

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/login");
    }
  }, [isLoading, isAuthenticated, router]);

  useEffect(() => {
    if (!token) return;
    setLoading(true);

    Promise.all([
      diagnosisApi.getHistory(token).catch(() => []),
      chatApi.listSessions(token).catch(() => []),
      hospitalApi.healthSummary(token).catch(() => null),
    ])
      .then(([diag, sess, summary]) => {
        setDiagnoses(diag);
        setSessions(sess);
        setHealthSummary(summary);
      })
      .finally(() => setLoading(false));
  }, [token]);

  // Computed chart data
  const triageDist = Object.entries(
    diagnoses.reduce<Record<string, number>>((acc, d) => {
      acc[d.triage_level] = (acc[d.triage_level] || 0) + 1;
      return acc;
    }, {})
  ).map(([name, value]) => ({ name, value }));

  const conditionFreq = Object.entries(
    diagnoses.reduce<Record<string, number>>((acc, d) => {
      acc[d.primary_condition] = (acc[d.primary_condition] || 0) + 1;
      return acc;
    }, {})
  )
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([condition, count]) => ({ condition, count }));

  const confidenceOverTime = diagnoses
    .slice(0, 20)
    .reverse()
    .map((d, i) => ({
      visit: i + 1,
      confidence: Math.round(d.primary_confidence * 100),
      condition: d.primary_condition,
    }));

  async function handleDrugSearch() {
    if (!drugSearch.trim() || !token) return;
    setDrugLoading(true);
    try {
      const res = await drugApi.search(drugSearch, token);
      setDrugResult(res);
    } catch (err) {
      console.error("Drug search failed:", err);
    } finally {
      setDrugLoading(false);
    }
  }

  if (isLoading || loading) {
    return (
      <div className="container mx-auto max-w-6xl p-4 space-y-4">
        <Skeleton className="h-8 w-48" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto max-w-6xl p-4">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold">Health Dashboard</h1>
        <div className="flex gap-1 rounded-lg border bg-muted p-1">
          {(["overview", "history", "drugs"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                activeTab === tab
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {tab === "overview" ? "📊 Overview" : tab === "history" ? "📋 History" : "💊 Drugs"}
            </button>
          ))}
        </div>
      </div>

      {/* ── Overview Tab ── */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          {/* Stat Cards */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              title="Total Consultations"
              value={sessions.length}
              icon="💬"
            />
            <StatCard
              title="Diagnoses"
              value={diagnoses.length}
              icon="🩺"
            />
            <StatCard
              title="Common Symptoms"
              value={healthSummary?.common_symptoms?.length || 0}
              icon="🫀"
            />
            <StatCard
              title="Health Score"
              value={healthSummary?.health_score ?? "—"}
              icon="📈"
            />
          </div>

          {/* Charts Row */}
          <div className="grid gap-6 md:grid-cols-2">
            {/* Triage Distribution Pie */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Triage Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                {triageDist.length > 0 ? (
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={triageDist}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        label={({ name, value }) => `${name}: ${value}`}
                      >
                        {triageDist.map((entry) => (
                          <Cell
                            key={entry.name}
                            fill={TRIAGE_COLORS[entry.name] || "#94a3b8"}
                          />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="py-16 text-center text-sm text-muted-foreground">
                    No diagnosis data yet
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Top Conditions Bar */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  Most Common Conditions
                </CardTitle>
              </CardHeader>
              <CardContent>
                {conditionFreq.length > 0 ? (
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={conditionFreq} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis
                        type="category"
                        dataKey="condition"
                        width={120}
                        tick={{ fontSize: 11 }}
                      />
                      <Tooltip />
                      <Bar dataKey="count" fill="hsl(221.2, 83.2%, 53.3%)" radius={4} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="py-16 text-center text-sm text-muted-foreground">
                    No diagnosis data yet
                  </p>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Confidence Over Time */}
          {confidenceOverTime.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  Diagnosis Confidence Over Time
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={confidenceOverTime}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="visit" label={{ value: "Visit #", position: "bottom" }} />
                    <YAxis domain={[0, 100]} unit="%" />
                    <Tooltip
                      formatter={(val: number) => `${val}%`}
                      labelFormatter={(v) => `Visit ${v}`}
                    />
                    <Line
                      type="monotone"
                      dataKey="confidence"
                      stroke="hsl(221.2, 83.2%, 53.3%)"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Health Summary */}
          {healthSummary && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Health Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm">{healthSummary.summary_text}</p>
                {healthSummary.common_symptoms?.length > 0 && (
                  <div>
                    <h4 className="text-sm font-semibold">Common Symptoms:</h4>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {healthSummary.common_symptoms.map((s: string) => (
                        <Badge key={s} variant="outline" className="text-xs">
                          {s.replace(/_/g, " ")}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* ── History Tab ── */}
      {activeTab === "history" && (
        <ScrollArea className="h-[calc(100vh-12rem)]">
          <div className="space-y-3">
            {diagnoses.length === 0 ? (
              <Card>
                <CardContent className="py-12 text-center text-muted-foreground">
                  <div className="mb-2 text-4xl">📋</div>
                  No diagnosis history yet. Use the Symptom Checker to get started.
                </CardContent>
              </Card>
            ) : (
              diagnoses.map((d) => (
                <Card key={d.id}>
                  <CardContent className="flex items-start justify-between p-4">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold">{d.primary_condition}</h3>
                        <Badge
                          variant={
                            d.triage_level === "EMERGENCY"
                              ? "emergency"
                              : d.triage_level === "URGENT"
                              ? "urgent"
                              : d.triage_level === "ROUTINE"
                              ? "routine"
                              : "self-care"
                          }
                        >
                          {d.triage_level}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Confidence: {(d.primary_confidence * 100).toFixed(0)}%
                        {d.icd10_codes?.length
                          ? ` · ICD-10: ${d.icd10_codes.join(", ")}`
                          : ""}
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {d.input_symptoms.slice(0, 5).map((s) => (
                          <Badge key={s} variant="outline" className="text-xs">
                            {s.replace(/_/g, " ")}
                          </Badge>
                        ))}
                        {d.input_symptoms.length > 5 && (
                          <Badge variant="outline" className="text-xs">
                            +{d.input_symptoms.length - 5} more
                          </Badge>
                        )}
                      </div>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {new Date(d.created_at).toLocaleDateString()}
                    </span>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </ScrollArea>
      )}

      {/* ── Drugs Tab ── */}
      {activeTab === "drugs" && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                💊 Drug Information Lookup (FDA)
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  placeholder="Enter drug name (e.g., aspirin, metformin)"
                  value={drugSearch}
                  onChange={(e) => setDrugSearch(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleDrugSearch()}
                />
                <Button onClick={handleDrugSearch} disabled={drugLoading}>
                  {drugLoading ? "Searching…" : "Search"}
                </Button>
              </div>

              {drugResult && (
                <div className="space-y-3">
                  <p className="text-sm text-muted-foreground">
                    Found {drugResult.meta?.total || 0} results for &quot;{drugResult.query}&quot;
                  </p>
                  {drugResult.results?.map((d: any, i: number) => (
                    <Card key={i} className="bg-muted/30">
                      <CardContent className="p-4 space-y-2">
                        <h3 className="font-semibold">
                          {d.brand_name || d.generic_name || "Unknown"}
                        </h3>
                        {d.generic_name && (
                          <p className="text-sm text-muted-foreground">
                            Generic: {d.generic_name}
                          </p>
                        )}
                        {d.manufacturer && (
                          <p className="text-xs text-muted-foreground">
                            Manufacturer: {d.manufacturer}
                          </p>
                        )}
                        {d.indications_and_usage && (
                          <div>
                            <h4 className="text-sm font-medium">Indications</h4>
                            <p className="text-xs text-muted-foreground">
                              {d.indications_and_usage}
                            </p>
                          </div>
                        )}
                        {d.warnings && (
                          <div>
                            <h4 className="text-sm font-medium text-destructive">
                              ⚠️ Warnings
                            </h4>
                            <p className="text-xs text-muted-foreground">
                              {d.warnings}
                            </p>
                          </div>
                        )}
                        {d.drug_interactions && (
                          <div>
                            <h4 className="text-sm font-medium">
                              Drug Interactions
                            </h4>
                            <p className="text-xs text-muted-foreground">
                              {d.drug_interactions}
                            </p>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

function StatCard({
  title,
  value,
  icon,
}: {
  title: string;
  value: string | number;
  icon: string;
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-4 p-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 text-2xl">
          {icon}
        </div>
        <div>
          <p className="text-sm text-muted-foreground">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
        </div>
      </CardContent>
    </Card>
  );
}
