"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/providers/auth-provider";
import { symptomApi } from "@/lib/api-client";
import { BodyMap, BODY_REGIONS } from "@/components/symptom-checker/body-map";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import ReactMarkdown from "react-markdown";
import { cn } from "@/lib/utils";

export default function SymptomCheckerPage() {
  const router = useRouter();
  const { token, isAuthenticated, isLoading } = useAuth();
  const [selectedRegions, setSelectedRegions] = useState<Set<string>>(new Set());
  const [selectedSymptoms, setSelectedSymptoms] = useState<Set<string>>(new Set());
  const [additionalText, setAdditionalText] = useState("");
  const [result, setResult] = useState<any>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [step, setStep] = useState<"select" | "review" | "result">("select");

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/login");
    }
  }, [isLoading, isAuthenticated, router]);

  function toggleRegion(regionId: string) {
    setSelectedRegions((prev) => {
      const next = new Set(prev);
      if (next.has(regionId)) {
        next.delete(regionId);
        // Remove symptoms from that region
        const region = BODY_REGIONS.find((r) => r.id === regionId);
        if (region) {
          setSelectedSymptoms((s) => {
            const ns = new Set(s);
            region.symptoms.forEach((sym) => ns.delete(sym));
            return ns;
          });
        }
      } else {
        next.add(regionId);
      }
      return next;
    });
  }

  function toggleSymptom(symptom: string) {
    setSelectedSymptoms((prev) => {
      const next = new Set(prev);
      if (next.has(symptom)) {
        next.delete(symptom);
      } else {
        next.add(symptom);
      }
      return next;
    });
  }

  // Build list of available symptoms based on selected regions
  const availableSymptoms = BODY_REGIONS
    .filter((r) => selectedRegions.has(r.id))
    .flatMap((r) => r.symptoms)
    .filter((v, i, a) => a.indexOf(v) === i); // deduplicate

  async function handleAnalyze() {
    if (!token) return;
    setAnalyzing(true);
    try {
      const symptomList = Array.from(selectedSymptoms);
      const text =
        symptomList.map((s) => s.replace(/_/g, " ")).join(", ") +
        (additionalText ? `. ${additionalText}` : "");

      const res = await symptomApi.diagnose(text, token);
      setResult(res);
      setStep("result");
    } catch (err) {
      console.error("Analysis failed:", err);
    } finally {
      setAnalyzing(false);
    }
  }

  function getTriageColor(level: string) {
    switch (level?.toLowerCase()) {
      case "emergency": return "bg-red-500 text-white";
      case "urgent": return "bg-orange-500 text-white";
      case "routine": return "bg-green-600 text-white";
      case "self_care": return "bg-blue-500 text-white";
      default: return "bg-muted";
    }
  }

  if (isLoading) {
    return (
      <div className="flex h-[calc(100vh-3.5rem)] items-center justify-center">
        <div className="animate-pulse text-muted-foreground">Loading…</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto max-w-6xl p-4">
      <h1 className="mb-6 text-2xl font-bold">Symptom Checker</h1>

      {/* Progress Steps */}
      <div className="mb-8 flex items-center justify-center gap-4">
        {(["select", "review", "result"] as const).map((s, i) => (
          <div key={s} className="flex items-center gap-2">
            <div
              className={cn(
                "flex h-8 w-8 items-center justify-center rounded-full text-sm font-bold",
                step === s
                  ? "bg-primary text-primary-foreground"
                  : i < ["select", "review", "result"].indexOf(step)
                  ? "bg-primary/20 text-primary"
                  : "bg-muted text-muted-foreground"
              )}
            >
              {i + 1}
            </div>
            <span className="hidden text-sm font-medium sm:inline">
              {s === "select" ? "Select" : s === "review" ? "Review" : "Results"}
            </span>
            {i < 2 && <div className="h-px w-8 bg-border" />}
          </div>
        ))}
      </div>

      {/* Step 1: Body Map Selection */}
      {step === "select" && (
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">
                Select affected body regions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <BodyMap
                selectedRegions={selectedRegions}
                onToggleRegion={toggleRegion}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">
                Select specific symptoms ({selectedSymptoms.size} selected)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[420px] pr-4">
                {availableSymptoms.length === 0 ? (
                  <p className="py-8 text-center text-sm text-muted-foreground">
                    Click a body region on the map to see available symptoms
                  </p>
                ) : (
                  <div className="space-y-1">
                    {availableSymptoms.map((symptom) => (
                      <label
                        key={symptom}
                        className={cn(
                          "flex cursor-pointer items-center gap-2 rounded-md px-3 py-2 transition-colors hover:bg-accent",
                          selectedSymptoms.has(symptom) && "bg-primary/10"
                        )}
                      >
                        <input
                          type="checkbox"
                          checked={selectedSymptoms.has(symptom)}
                          onChange={() => toggleSymptom(symptom)}
                          className="rounded border-input"
                        />
                        <span className="text-sm">
                          {symptom.replace(/_/g, " ")}
                        </span>
                      </label>
                    ))}
                  </div>
                )}
              </ScrollArea>

              <div className="mt-4 space-y-3">
                <Textarea
                  placeholder="Additional details (duration, severity, etc.)"
                  value={additionalText}
                  onChange={(e) => setAdditionalText(e.target.value)}
                  rows={3}
                />
                <Button
                  className="w-full"
                  onClick={() => setStep("review")}
                  disabled={selectedSymptoms.size === 0}
                >
                  Review Symptoms →
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Step 2: Review */}
      {step === "review" && (
        <Card className="mx-auto max-w-2xl">
          <CardHeader>
            <CardTitle>Review Your Symptoms</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {Array.from(selectedSymptoms).map((s) => (
                <Badge key={s} variant="secondary" className="text-sm">
                  {s.replace(/_/g, " ")}
                  <button
                    className="ml-1 text-xs hover:text-destructive"
                    onClick={() => toggleSymptom(s)}
                  >
                    ✕
                  </button>
                </Badge>
              ))}
            </div>

            {additionalText && (
              <div className="rounded-md bg-muted p-3 text-sm">
                <strong>Additional notes:</strong> {additionalText}
              </div>
            )}

            <div className="flex gap-3">
              <Button variant="outline" onClick={() => setStep("select")}>
                ← Back
              </Button>
              <Button
                className="flex-1"
                onClick={handleAnalyze}
                disabled={analyzing}
              >
                {analyzing ? "Analyzing…" : "🔍 Analyze Symptoms"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 3: Results */}
      {step === "result" && result && (
        <div className="mx-auto max-w-3xl space-y-6">
          {/* Triage Banner */}
          {result.triage_result && (
            <Card
              className={cn(
                "border-2",
                result.triage_result.triage_level === "EMERGENCY"
                  ? "border-red-500"
                  : result.triage_result.triage_level === "URGENT"
                  ? "border-orange-500"
                  : "border-green-500"
              )}
            >
              <CardContent className="flex items-center gap-4 p-6">
                <div
                  className={cn(
                    "flex h-14 w-14 items-center justify-center rounded-full text-2xl",
                    getTriageColor(result.triage_result.triage_level)
                  )}
                >
                  {result.triage_result.triage_level === "EMERGENCY"
                    ? "🚨"
                    : result.triage_result.triage_level === "URGENT"
                    ? "⚠️"
                    : "✅"}
                </div>
                <div>
                  <h2 className="text-xl font-bold">
                    Triage: {result.triage_result.triage_level}
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    {result.triage_result.reasoning || "Based on symptom analysis"}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Diagnosis */}
          {result.diagnosis_result?.primary_diagnosis && (
            <Card>
              <CardHeader>
                <CardTitle>Possible Diagnosis</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold">
                    {result.diagnosis_result.primary_diagnosis.condition}
                  </h3>
                  <Badge>
                    {(
                      (result.diagnosis_result.primary_diagnosis.confidence || 0) *
                      100
                    ).toFixed(0)}
                    % confidence
                  </Badge>
                </div>
                {result.diagnosis_result.primary_diagnosis.explanation && (
                  <p className="text-sm text-muted-foreground">
                    {result.diagnosis_result.primary_diagnosis.explanation}
                  </p>
                )}

                {/* Differentials */}
                {result.diagnosis_result.differential_diagnoses?.length > 0 && (
                  <div className="mt-3">
                    <h4 className="text-sm font-semibold">Other possibilities:</h4>
                    <div className="mt-1 space-y-1">
                      {result.diagnosis_result.differential_diagnoses.map(
                        (d: any, i: number) => (
                          <div
                            key={i}
                            className="flex items-center justify-between text-sm"
                          >
                            <span>{d.condition}</span>
                            <span className="text-muted-foreground">
                              {((d.confidence || 0) * 100).toFixed(0)}%
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Precautions */}
          {result.precaution_result && (
            <Card>
              <CardHeader>
                <CardTitle>Recommended Actions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  {result.precaution_result.immediate_actions && (
                    <div>
                      <h4>Immediate Actions</h4>
                      <ul>
                        {result.precaution_result.immediate_actions.map(
                          (a: any, i: number) => (
                            <li key={i}>{a.action || a}</li>
                          )
                        )}
                      </ul>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Summary */}
          {result.response && (
            <Card>
              <CardHeader>
                <CardTitle>Full Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <ReactMarkdown>{result.response}</ReactMarkdown>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Disclaimer */}
          <div className="rounded-lg border border-orange-200 bg-orange-50 p-4 text-sm text-orange-800 dark:border-orange-900 dark:bg-orange-950 dark:text-orange-200">
            ⚠️ This analysis is AI-generated and for informational purposes only.
            Always consult a qualified healthcare professional for medical advice.
          </div>

          <div className="flex gap-3">
            <Button
              variant="outline"
              onClick={() => {
                setStep("select");
                setResult(null);
                setSelectedRegions(new Set());
                setSelectedSymptoms(new Set());
                setAdditionalText("");
              }}
            >
              Start Over
            </Button>
            <Button onClick={() => router.push("/chat")}>
              💬 Discuss with AI Chat
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
