import Link from "next/link";

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-slate-900 dark:to-slate-800">
      <div className="mx-auto max-w-4xl px-6 text-center">
        {/* Logo / Title */}
        <div className="mb-8">
          <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl bg-primary text-3xl text-primary-foreground shadow-lg">
            🏥
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-foreground sm:text-6xl">
            Healthcare AI
            <span className="text-primary"> Platform</span>
          </h1>
          <p className="mt-4 text-lg text-muted-foreground">
            AI-powered symptom analysis, disease prediction, and personalised
            health recommendations — all in one platform.
          </p>
        </div>

        {/* Feature Cards */}
        <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          <FeatureCard
            icon="💬"
            title="AI Chat"
            description="Describe your symptoms in natural language and get instant analysis."
            href="/chat"
          />
          <FeatureCard
            icon="🫀"
            title="Body Map"
            description="Select symptoms on an interactive body map for precise input."
            href="/symptom-checker"
          />
          <FeatureCard
            icon="📊"
            title="Health Dashboard"
            description="Track your health trends, diagnosis history, and insights."
            href="/dashboard"
          />
        </div>

        {/* Disclaimer */}
        <p className="mt-12 text-xs text-muted-foreground">
          ⚠️ This platform is for informational purposes only and is not a
          substitute for professional medical advice, diagnosis, or treatment.
          Always consult a qualified healthcare provider.
        </p>
      </div>
    </main>
  );
}

function FeatureCard({
  icon,
  title,
  description,
  href,
}: {
  icon: string;
  title: string;
  description: string;
  href: string;
}) {
  return (
    <Link
      href={href}
      className="group rounded-xl border bg-card p-6 text-left shadow-sm transition-all hover:shadow-md hover:-translate-y-1"
    >
      <span className="text-3xl">{icon}</span>
      <h3 className="mt-3 text-lg font-semibold text-card-foreground group-hover:text-primary">
        {title}
      </h3>
      <p className="mt-1 text-sm text-muted-foreground">{description}</p>
    </Link>
  );
}
