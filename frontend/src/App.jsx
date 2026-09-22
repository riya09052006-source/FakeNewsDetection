import {
  useEffect,
  useState,
} from "react";

import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  FileText,
  Loader2,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";

import {
  checkBackend,
  getModelInfo,
  predictNews,
} from "./services/api";

const SAMPLE_ARTICLES = [
  {
    title: "Scientific Research",
    text: `Researchers published a detailed scientific report after
conducting multiple experiments. The researchers explained that
additional independent studies are necessary to confirm the
findings and understand their broader implications.`,
  },

  {
    title: "Government Report",
    text: `The national statistics agency released its latest economic
report containing updated employment and inflation figures based
on officially collected data and publicly documented methods.`,
  },

  {
    title: "Viral Claim",
    text: `A viral social media post claims that a secret drink can
instantly cure every known disease. The post provides no
scientific evidence, medical study, or credible source to
support the claim.`,
  },
];

function App() {

  const [newsText, setNewsText] =
    useState("");

  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [backendStatus, setBackendStatus] =
    useState("checking");

  const [modelInfo, setModelInfo] =
    useState(null);


  async function refreshSystemStatus() {

    try {

      const health =
        await checkBackend();

      setBackendStatus(
        health.model_ready
          ? "online"
          : "degraded"
      );

    } catch {

      setBackendStatus(
        "offline"
      );
    }
  }


  useEffect(() => {

    async function loadSystemInfo() {

      try {

        const health =
          await checkBackend();

        const info =
          await getModelInfo();

        setBackendStatus(
          health.model_ready
            ? "online"
            : "degraded"
        );

        setModelInfo(info);

      } catch {

        setBackendStatus(
          "offline"
        );
      }
    }

    loadSystemInfo();

  }, []);


  async function handleAnalyze() {

    setError("");
    setResult(null);

    const text =
      newsText.trim();

    if (!text) {

      setError(
        "Please enter a news article before analyzing."
      );

      return;
    }

    if (text.length < 20) {

      setError(
        "Please enter at least 20 characters."
      );

      return;
    }

    if (text.length > 100000) {

      setError(
        "The article is too long. Maximum length is 100,000 characters."
      );

      return;
    }

    setLoading(true);

    try {

      const data =
        await predictNews(text);

      setResult(data);

    } catch (err) {

      console.error(
        "Prediction error:",
        err
      );

      const detail =
        err.response?.data?.detail;

      setError(
        detail ||
        "The prediction service is currently unavailable. Please check that the backend is running."
      );

    } finally {

      setLoading(false);
    }
  }


  function handleClear() {

    setNewsText("");
    setResult(null);
    setError("");
  }


  function handleTextareaKeyDown(event) {

    if (
      event.ctrlKey &&
      event.key === "Enter"
    ) {

      event.preventDefault();

      handleAnalyze();
    }
  }


  function getScoreDirection(score) {

    if (score > 0) {

      return {
        text: "Model leans toward REAL",
        className: "text-emerald-400",
      };

    }

    return {
      text: "Model leans toward FAKE",
      className: "text-red-400",
    };
  }


  const isReal =
    result?.prediction === "REAL";

  const scoreDirection =
    result
      ? getScoreDirection(
          result.decision_score
        )
      : null;


  return (
    <div className="min-h-screen text-white">

      {/* NAVBAR */}

      <header className="border-b border-white/10 bg-black/20 backdrop-blur-xl">

        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">

          <div className="flex items-center gap-3">

            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500/20">

              <ShieldCheck
                className="text-blue-400"
                size={23}
              />

            </div>

            <div>

              <h1 className="text-lg font-bold tracking-tight">
                FakeGuard AI
              </h1>

              <p className="text-xs text-gray-500">
                News Classification System
              </p>

            </div>

          </div>


          <div className="flex items-center gap-2">

            <span
              className={`h-2.5 w-2.5 rounded-full ${
                backendStatus === "online"
                  ? "bg-emerald-400"
                  : backendStatus === "checking"
                  ? "bg-yellow-400"
                  : "bg-red-400"
              }`}
            />

            <span className="text-sm text-gray-400">

              {backendStatus === "online"
                ? "System Online"
                : backendStatus === "checking"
                ? "Checking..."
                : "Backend Offline"}

            </span>

          </div>

        </div>

      </header>


      {/* HERO */}

      <main className="mx-auto max-w-7xl px-4 py-10 sm:px-6 sm:py-16">

        <section className="mx-auto max-w-4xl text-center">

          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-400/10 px-4 py-2 text-sm text-blue-300">

            <Sparkles size={16} />

            AI-Powered News Analysis

          </div>


          <h2 className="text-4xl font-black tracking-tight sm:text-6xl">

            Detect suspicious news
            <span className="block bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">

              with machine learning.

            </span>

          </h2>


          <p className="mx-auto mt-6 max-w-2xl text-base leading-7 text-gray-400">

            Analyze a news article using a trained
            machine-learning classification model
            powered by Word and Character TF-IDF
            features with an optimized Linear SVM.

          </p>

        </section>


        {/* INPUT CARD */}

        <section className="mx-auto mt-12 max-w-4xl">

          <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-5 shadow-2xl shadow-black/30 backdrop-blur-xl sm:p-7">

            <div className="mb-5 flex items-center justify-between">

              <div className="flex items-center gap-3">

                <div className="rounded-xl bg-white/10 p-2">

                  <FileText
                    size={20}
                    className="text-blue-300"
                  />

                </div>

                <div>

                  <h3 className="font-semibold">
                    News Article
                  </h3>

                  <p className="text-xs text-gray-500">
                    Paste the article you want to analyze
                  </p>

                </div>

              </div>


              <div className="text-right">

                <p
                  className={`text-xs ${
                    newsText.length > 90000
                      ? "text-yellow-400"
                      : "text-gray-500"
                  }`}
                >

                  {newsText.length.toLocaleString()} / 100,000

                </p>

              </div>

            </div>


            {/* SAMPLE ARTICLES */}

            <div className="mb-4">

              <div className="mb-2 flex items-center justify-between">

                <p className="text-xs font-medium text-gray-500">
                  Quick examples
                </p>

                <p className="text-xs text-gray-600">
                  Try a sample
                </p>

              </div>

              <div className="flex flex-wrap gap-2">

                {SAMPLE_ARTICLES.map(
                  (article) => (

                    <button
                      key={article.title}
                      onClick={() =>
                        setNewsText(article.text)
                      }
                      className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-gray-400 transition hover:border-blue-400/30 hover:bg-blue-400/10 hover:text-blue-300"
                    >
                      {article.title}
                    </button>

                  )
                )}

              </div>

            </div>


            <textarea
              value={newsText}
              onChange={(event) =>
                setNewsText(
                  event.target.value
                )
              }
              onKeyDown={handleTextareaKeyDown}
              placeholder="Paste a news article here... (Ctrl + Enter to analyze)"
              maxLength={100000}
              rows={10}
              className="w-full resize-none rounded-2xl border border-white/10 bg-black/20 p-5 text-sm leading-7 text-gray-200 outline-none transition placeholder:text-gray-600 focus:border-blue-400/50 focus:ring-2 focus:ring-blue-400/10"
            />


            {/* MINIMUM TEXT PROGRESS INDICATOR */}

            <div className="mt-2 h-1 overflow-hidden rounded-full bg-white/5">

              <div
                className="h-full rounded-full bg-blue-400 transition-all"
                style={{
                  width: `${Math.min(
                    (newsText.length / 500) * 100,
                    100
                  )}%`,
                }}
              />

            </div>


            {error && (

              <div className="mt-4 flex items-start gap-3 rounded-xl border border-red-400/20 bg-red-400/10 p-4 text-sm text-red-300">

                <AlertCircle
                  size={18}
                  className="mt-0.5 shrink-0"
                />

                <span>
                  {error}
                </span>

              </div>

            )}


            <div className="mt-5 flex flex-col gap-3 sm:flex-row">

              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-blue-500 px-6 py-3.5 font-semibold text-white transition hover:bg-blue-400 disabled:cursor-not-allowed disabled:opacity-60"
              >

                {loading ? (

                  <>
                    <Loader2
                      size={18}
                      className="animate-spin"
                    />

                    Analyzing article...

                  </>

                ) : (

                  <>
                    <Sparkles size={18} />

                    Analyze News
                  </>

                )}

              </button>


              <button
                onClick={handleClear}
                className="rounded-xl border border-white/10 px-6 py-3.5 font-semibold text-gray-300 transition hover:bg-white/5"
              >

                Clear

              </button>

            </div>


            {loading && (

              <p className="mt-3 text-center text-xs text-gray-600">

                Running local TF-IDF feature extraction
                and SVM inference...

              </p>

            )}

          </div>

        </section>


        {/* RESULT */}

        {result && (

          <section className="mx-auto mt-8 max-w-4xl">

            <div
              className={`rounded-3xl border p-7 backdrop-blur-xl ${
                isReal
                  ? "border-emerald-400/20 bg-emerald-400/[0.06]"
                  : "border-red-400/20 bg-red-400/[0.06]"
              }`}
            >

              <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">

                <div className="flex items-center gap-4">

                  <div
                    className={`flex h-14 w-14 items-center justify-center rounded-2xl ${
                      isReal
                        ? "bg-emerald-400/15"
                        : "bg-red-400/15"
                    }`}
                  >

                    {isReal ? (

                      <CheckCircle2
                        size={28}
                        className="text-emerald-400"
                      />

                    ) : (

                      <XCircle
                        size={28}
                        className="text-red-400"
                      />

                    )}

                  </div>


                  <div>

                    <p className="text-sm text-gray-500">
                      Model Prediction
                    </p>

                    <h3
                      className={`mt-1 text-3xl font-black ${
                        isReal
                          ? "text-emerald-400"
                          : "text-red-400"
                      }`}
                    >

                      {isReal
                        ? "LIKELY REAL"
                        : "LIKELY FAKE"}

                    </h3>

                  </div>

                </div>


                <div className="rounded-2xl bg-black/20 px-5 py-4">

                  <p className="text-xs text-gray-500">
                    Decision Score
                  </p>

                  <p className="mt-1 font-mono text-lg">
                    {result.decision_score.toFixed(4)}
                  </p>

                  <p
                    className={`mt-1 text-xs ${scoreDirection.className}`}
                  >
                    {scoreDirection.text}
                  </p>

                </div>

              </div>


              <div className="mt-7 grid gap-3 sm:grid-cols-2">

                <div className="rounded-2xl border border-white/5 bg-black/20 p-4">

                  <p className="text-xs text-gray-500">
                    Model
                  </p>

                  <p className="mt-1 text-sm font-medium">
                    {result.model}
                  </p>

                </div>


                <div className="rounded-2xl border border-white/5 bg-black/20 p-4">

                  <p className="text-xs text-gray-500">
                    Features
                  </p>

                  <p className="mt-1 text-sm font-medium">
                    {result.feature_type}
                  </p>

                </div>

              </div>


              {/* MODEL TRANSPARENCY */}

              <div className="mt-5 grid gap-3 sm:grid-cols-3">

                <div className="rounded-2xl border border-white/5 bg-black/20 p-4">

                  <p className="text-xs text-gray-500">
                    Class
                  </p>

                  <p className="mt-1 text-sm font-semibold">
                    {result.label_id === 1
                      ? "REAL (1)"
                      : "FAKE (0)"}
                  </p>

                </div>


                <div className="rounded-2xl border border-white/5 bg-black/20 p-4">

                  <p className="text-xs text-gray-500">
                    Algorithm
                  </p>

                  <p className="mt-1 text-sm font-semibold">
                    Linear SVM
                  </p>

                </div>


                <div className="rounded-2xl border border-white/5 bg-black/20 p-4">

                  <p className="text-xs text-gray-500">
                    Processing
                  </p>

                  <p className="mt-1 text-sm font-semibold">
                    Local
                  </p>

                </div>

              </div>


              {/* WARNING */}

              <div className="mt-5 flex items-start gap-3 rounded-2xl border border-yellow-400/10 bg-yellow-400/5 p-4 text-xs leading-6 text-gray-400">

                <AlertCircle
                  size={17}
                  className="mt-1 shrink-0 text-yellow-400"
                />

                <p>

                  This is a machine-learning
                  classification based on learned
                  patterns in the training data.
                  It is not independent factual
                  verification of the article.

                </p>

              </div>


              {/* REQUEST ID */}

              {result.request_id && (

                <p className="mt-4 text-right font-mono text-[10px] text-gray-700">

                  Request ID: {result.request_id}

                </p>

              )}

            </div>

          </section>

        )}


        {/* SYSTEM INFORMATION */}

        <section className="mx-auto mt-12 grid max-w-4xl gap-4 sm:grid-cols-3">

          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">

            <Cpu
              size={22}
              className="text-blue-400"
            />

            <p className="mt-4 text-xs text-gray-500">
              Model
            </p>

            <p className="mt-1 text-sm font-semibold">
              {modelInfo?.model ||
                "Optimized Linear SVM"}
            </p>

          </div>


          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">

            <FileText
              size={22}
              className="text-purple-400"
            />

            <p className="mt-4 text-xs text-gray-500">
              Feature Engineering
            </p>

            <p className="mt-1 text-sm font-semibold">
              Word + Character TF-IDF
            </p>

          </div>


          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">

            <ShieldCheck
              size={22}
              className="text-emerald-400"
            />

            <p className="mt-4 text-xs text-gray-500">
              Processing
            </p>

            <p className="mt-1 text-sm font-semibold">
              Local ML Inference
            </p>

          </div>

        </section>


        {/* HOW IT WORKS */}

        <section className="mx-auto mt-16 max-w-4xl">

          <div className="mb-6 text-center">

            <p className="text-sm font-semibold text-blue-400">
              HOW IT WORKS
            </p>

            <h3 className="mt-2 text-2xl font-bold">
              From article to prediction
            </h3>

          </div>


          <div className="grid gap-4 sm:grid-cols-3">

            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">

              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-blue-400/10 text-blue-400">

                01

              </div>

              <h4 className="font-semibold">
                Text Processing
              </h4>

              <p className="mt-2 text-sm leading-6 text-gray-500">

                The article is normalized and cleaned
                before feature extraction.

              </p>

            </div>


            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">

              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-purple-400/10 text-purple-400">

                02

              </div>

              <h4 className="font-semibold">
                Feature Extraction
              </h4>

              <p className="mt-2 text-sm leading-6 text-gray-500">

                Word and character TF-IDF features
                represent the article for the model.

              </p>

            </div>


            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">

              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-400/10 text-emerald-400">

                03

              </div>

              <h4 className="font-semibold">
                ML Classification
              </h4>

              <p className="mt-2 text-sm leading-6 text-gray-500">

                The optimized Linear SVM produces
                the final model classification.

              </p>

            </div>

          </div>

        </section>


        {/* FOOTER */}

        <footer className="mx-auto mt-20 max-w-4xl border-t border-white/10 py-8 text-center">

          <div className="flex items-center justify-center gap-2">

            <ShieldCheck
              size={16}
              className="text-blue-400"
            />

            <span className="text-sm font-semibold text-gray-400">
              FakeGuard AI
            </span>

          </div>

          <p className="mt-2 text-xs leading-6 text-gray-600">

            Fake News Detection Using Machine Learning

            <br />

            Optimized Linear SVM · Word + Character TF-IDF

          </p>

          <p className="mt-3 text-[10px] text-gray-700">

            Academic project · Model predictions are not
            independent fact verification.

          </p>

        </footer>

      </main>

    </div>
  );
}


export default App;
