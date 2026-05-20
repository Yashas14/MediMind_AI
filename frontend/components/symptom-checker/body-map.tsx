"use client";

/**
 * SVG interactive body map for symptom selection.
 *
 * Each body region is clickable and highlights on hover/select.
 * Maps body regions to symptom categories from the 132-symptom dataset.
 */

import { cn } from "@/lib/utils";

interface BodyRegion {
  id: string;
  label: string;
  symptoms: string[];
  path: string; // SVG path data
}

const BODY_REGIONS: BodyRegion[] = [
  {
    id: "head",
    label: "Head",
    symptoms: ["headache", "dizziness", "altered_sensorium", "visual_disturbances", "blurred_and_distorted_vision", "lack_of_concentration", "slurred_speech"],
    path: "M 150 30 C 175 15 225 15 250 30 C 265 45 270 75 260 95 C 250 115 235 120 200 125 C 165 120 150 115 140 95 C 130 75 135 45 150 30 Z",
  },
  {
    id: "throat",
    label: "Throat / Neck",
    symptoms: ["patches_in_throat", "throat_irritation", "continuous_sneezing", "runny_nose", "congestion", "sinus_pressure", "loss_of_smell"],
    path: "M 180 125 L 220 125 L 225 155 L 175 155 Z",
  },
  {
    id: "chest",
    label: "Chest",
    symptoms: ["chest_pain", "breathlessness", "fast_heart_rate", "palpitations", "cough", "phlegm", "mucoid_sputum", "rusty_sputum", "blood_in_sputum"],
    path: "M 140 155 L 260 155 L 270 240 L 200 250 L 130 240 Z",
  },
  {
    id: "abdomen",
    label: "Abdomen",
    symptoms: ["stomach_pain", "acidity", "vomiting", "nausea", "loss_of_appetite", "indigestion", "abdominal_pain", "belly_pain", "constipation", "diarrhoea", "passage_of_gases", "stomach_bleeding"],
    path: "M 135 240 L 265 240 L 260 340 L 200 350 L 140 340 Z",
  },
  {
    id: "left_arm",
    label: "Left Arm",
    symptoms: ["joint_pain", "muscle_wasting", "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking"],
    path: "M 130 160 L 140 160 L 135 240 L 100 320 L 80 320 L 120 240 Z",
  },
  {
    id: "right_arm",
    label: "Right Arm",
    symptoms: ["joint_pain", "muscle_wasting", "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking"],
    path: "M 260 160 L 270 160 L 280 240 L 320 320 L 300 320 L 265 240 Z",
  },
  {
    id: "skin",
    label: "Skin (General)",
    symptoms: ["itching", "skin_rash", "nodal_skin_eruptions", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "blister", "red_sore_around_nose", "yellow_crust_ooze", "dischromic_patches"],
    path: "M 330 60 L 380 60 L 380 120 L 330 120 Z",
  },
  {
    id: "legs",
    label: "Legs",
    symptoms: ["knee_pain", "hip_joint_pain", "weakness_in_limbs", "painful_walking", "prominent_veins_on_calf", "swollen_legs", "swelling_of_stomach", "muscle_pain", "cramps"],
    path: "M 155 350 L 195 350 L 190 480 L 160 480 Z M 205 350 L 245 350 L 240 480 L 210 480 Z",
  },
  {
    id: "systemic",
    label: "Whole Body / Systemic",
    symptoms: ["fatigue", "high_fever", "mild_fever", "malaise", "lethargy", "sweating", "dehydration", "weight_loss", "weight_gain", "restlessness", "chills", "shivering", "anxiety", "mood_swings", "cold_hands_and_feets"],
    path: "M 330 140 L 380 140 L 380 200 L 330 200 Z",
  },
];

interface BodyMapProps {
  selectedRegions: Set<string>;
  onToggleRegion: (regionId: string) => void;
}

export function BodyMap({ selectedRegions, onToggleRegion }: BodyMapProps) {
  return (
    <div className="relative">
      <svg
        viewBox="0 0 400 510"
        className="mx-auto h-[500px] w-auto"
        role="img"
        aria-label="Interactive body map for symptom selection"
      >
        {/* Background body outline */}
        <path
          d="M 150 30 C 175 15 225 15 250 30 C 265 45 270 75 260 95 C 250 115 235 120 200 125 C 165 120 150 115 140 95 C 130 75 135 45 150 30 Z M 180 125 L 220 125 L 260 155 L 270 160 L 280 240 L 320 320 L 300 320 L 265 240 L 260 340 L 245 350 L 240 480 L 210 480 L 205 350 L 200 350 L 195 350 L 190 480 L 160 480 L 155 350 L 140 340 L 135 240 L 100 320 L 80 320 L 120 240 L 130 160 L 140 155 Z"
          fill="none"
          stroke="currentColor"
          strokeWidth="1"
          opacity={0.2}
        />

        {/* Interactive regions */}
        {BODY_REGIONS.map((region) => (
          <g key={region.id}>
            <path
              d={region.path}
              className={cn(
                "cursor-pointer stroke-2 transition-all duration-200",
                selectedRegions.has(region.id)
                  ? "fill-primary/40 stroke-primary"
                  : "fill-transparent stroke-muted-foreground/30 hover:fill-primary/20 hover:stroke-primary/60"
              )}
              onClick={() => onToggleRegion(region.id)}
              role="button"
              aria-label={`Select ${region.label} region`}
              aria-pressed={selectedRegions.has(region.id)}
            />
          </g>
        ))}

        {/* Region labels */}
        {BODY_REGIONS.map((region) => {
          const labelPos = getLabelPosition(region.id);
          return (
            <text
              key={`label-${region.id}`}
              x={labelPos.x}
              y={labelPos.y}
              textAnchor="middle"
              className="pointer-events-none select-none fill-foreground text-[10px] font-medium"
            >
              {region.label}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

function getLabelPosition(id: string): { x: number; y: number } {
  const positions: Record<string, { x: number; y: number }> = {
    head: { x: 200, y: 70 },
    throat: { x: 200, y: 140 },
    chest: { x: 200, y: 200 },
    abdomen: { x: 200, y: 290 },
    left_arm: { x: 95, y: 240 },
    right_arm: { x: 305, y: 240 },
    skin: { x: 355, y: 95 },
    legs: { x: 200, y: 420 },
    systemic: { x: 355, y: 175 },
  };
  return positions[id] || { x: 200, y: 200 };
}

export { BODY_REGIONS, type BodyRegion };
