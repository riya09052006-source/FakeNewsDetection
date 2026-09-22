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
  getHealth,
  getModelInfo,
  predictNews,
} from "./services/api";


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


  useEffect(() => {

    async function loadSystemInfo() {

      try {

        const health =
          await getHealth();

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
        "Please enter a news article."
      );

      return;
    }

    if (text.length < 20) {

      setError(
        "Please enter at least 20 characters."
      );

      return;
    }

    setLoading(true);

    try {

      const data =
        await predictNews(text);

      setResult(data);

    } catch (err) {

      if (
        err.response?.data?.detail
      ) {

        setError(
          err.response.data.detail
        );

      } else {

        setError(
          "Unable to connect to the prediction server."
        );
      }

    } finally {

      setLoading(false);
    }
  }


  function handleClear() {

    setNewsText("");
    setResult(null);
    setError("");
  }


  const isReal =
    result?.prediction === "REAL";


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

      <main className="mx-auto max-w-7xl px-6 py-16">

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


              <span className="text-xs text-gray-500">

                {newsText.length.toLocaleString()} chars

              </span>

            </div>


            <textarea
              value={newsText}
              onChange={(event) =>
                setNewsText(
                  event.target.value
                )
              }
              placeholder="Paste a news article here..."
              maxLength={100000}
              rows={10}
              className="w-full resize-none rounded-2xl border border-white/10 bg-black/20 p-5 text-sm leading-7 text-gray-200 outline-none transition placeholder:text-gray-600 focus:border-blue-400/50 focus:ring-2 focus:ring-blue-400/10"
            />


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

                    Analyzing...

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

                    {result.decision_score.toFixed(
                      4
                    )}

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


        {/* FOOTER */}

        <footer className="mx-auto mt-16 max-w-4xl border-t border-white/10 py-8 text-center">

          <p className="text-xs leading-6 text-gray-600">

            FakeGuard AI · Fake News Detection
            Using Machine Learning

            <br />

            Built for academic and research use.

          </p>

        </footer>

      </main>

    </div>
  );
}


export default App;
