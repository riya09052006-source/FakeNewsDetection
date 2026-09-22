import {
  useEffect,
  useState,
} from "react";

import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Cpu,
  FileText,
  History,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Trash2,
  XCircle,
} from "lucide-react";

import {
  checkBackend,
  deletePredictionHistory,
  explainNews,
  getAnalytics,
  getModelInfo,
  getPredictionHistory,
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

  const [explanation, setExplanation] =
    useState(null);

  const [explainLoading, setExplainLoading] =
    useState(false);

  const [history, setHistory] =
    useState([]);

  const [analytics, setAnalytics] =
    useState(null);

  const [historyLoading, setHistoryLoading] =
    useState(false);


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


  const loadAnalytics = async () => {

    try {

      setHistoryLoading(true);

      const [historyData, analyticsData] =
        await Promise.all([
          getPredictionHistory(),
          getAnalytics(),
        ]);

      setHistory(historyData.items || []);
      setAnalytics(analyticsData);

    } catch (err) {

      console.error(
        "Failed to load analytics:",
        err
      );

    } finally {

      setHistoryLoading(false);
    }
  };


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
    loadAnalytics();

  }, []);


  async function handleAnalyze() {

    setError("");
    setResult(null);
    setExplanation(null);

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

      await loadAnalytics();

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


  const handleExplain = async () => {

    if (!newsText.trim()) {
      return;
    }

    if (newsText.trim().length < 20) {
      return;
    }

    try {

      setExplainLoading(true);

      const explanationResult =
        await explainNews(newsText);

      setExplanation(explanationResult);

    } catch (err) {

      console.error(
        "Explanation error:",
        err
      );

      setError(
        "Unable to generate the model explanation."
      );

    } finally {

      setExplainLoading(false);
    }
  };


  function handleClear() {

    setNewsText("");
    setResult(null);
    setError("");
    setExplanation(null);
  }


  const handleClearHistory = async () => {

    const confirmed = window.confirm(
      "Are you sure you want to delete all prediction history?"
    );

    if (!confirmed) {
      return;
    }

    try {

      await deletePredictionHistory();
      await loadAnalytics();

    } catch (err) {

      console.error(
        "Failed to clear history:",
        err
      );
    }
  };


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


            {/* EXPLAIN BUTTON */}

            <div className="mt-6">

              <button
                onClick={handleExplain}
                disabled={explainLoading}
                className="w-full rounded-2xl border border-white/10 bg-white/5 px-5 py-4 text-sm font-semibold text-white transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
              >

                {explainLoading
                  ? "Generating explanation..."
                  : "Why did the model decide this?"}

              </button>

            </div>


            {/* EXPLANATION RESULTS */}

            {explanation && (

              <section className="mt-6 rounded-3xl border border-white/10 bg-white/[0.04] p-6 shadow-2xl">

                <div className="mb-6">

                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-blue-400">
                    Explainable AI
                  </p>

                  <h3 className="mt-2 text-2xl font-bold text-white">
                    Why did the model decide this?
                  </h3>

                  <p className="mt-2 text-sm leading-6 text-slate-400">
                    These features influenced the machine-learning model's
                    decision. They are not proof that the article is
                    factually true or false.
                  </p>

                </div>


                <div className="grid gap-6 md:grid-cols-2">

                  {/* REAL SIGNALS */}

                  <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/5 p-5">

                    <div className="mb-4">

                      <h4 className="font-semibold text-emerald-300">
                        Signals toward REAL
                      </h4>

                      <p className="mt-1 text-xs text-slate-500">
                        Positive model contributions
                      </p>

                    </div>


                    <div className="space-y-3">

                      {explanation.top_real_features.length > 0 ? (

                        explanation.top_real_features.map(
                          (item, index) => (

                            <div
                              key={`${item.feature}-${index}`}
                              className="flex items-center justify-between gap-4 rounded-xl bg-white/5 px-4 py-3"
                            >

                              <span className="break-all text-sm text-slate-200">
                                {item.feature}
                              </span>

                              <span className="shrink-0 text-sm font-semibold text-emerald-400">
                                +{item.contribution.toFixed(4)}
                              </span>

                            </div>

                          )
                        )

                      ) : (

                        <p className="text-sm text-slate-500">
                          No strong positive signals were found.
                        </p>

                      )}

                    </div>

                  </div>


                  {/* FAKE SIGNALS */}

                  <div className="rounded-2xl border border-red-400/20 bg-red-400/5 p-5">

                    <div className="mb-4">

                      <h4 className="font-semibold text-red-300">
                        Signals toward FAKE
                      </h4>

                      <p className="mt-1 text-xs text-slate-500">
                        Negative model contributions
                      </p>

                    </div>


                    <div className="space-y-3">

                      {explanation.top_fake_features.length > 0 ? (

                        explanation.top_fake_features.map(
                          (item, index) => (

                            <div
                              key={`${item.feature}-${index}`}
                              className="flex items-center justify-between gap-4 rounded-xl bg-white/5 px-4 py-3"
                            >

                              <span className="break-all text-sm text-slate-200">
                                {item.feature}
                              </span>

                              <span className="shrink-0 text-sm font-semibold text-red-400">
                                {item.contribution.toFixed(4)}
                              </span>

                            </div>

                          )
                        )

                      ) : (

                        <p className="text-sm text-slate-500">
                          No strong negative signals were found.
                        </p>

                      )}

                    </div>

                  </div>

                </div>


                <div className="mt-6 rounded-2xl border border-blue-400/10 bg-blue-400/5 p-4">

                  <p className="text-xs leading-6 text-slate-400">

                    <span className="font-semibold text-blue-300">
                      Important:
                    </span>{" "}

                    {explanation.explanation_note}

                  </p>

                </div>

              </section>

            )}

          </section>

        )}


        {/* ANALYTICS SECTION */}

        {analytics && (

          <section className="mx-auto mt-12 max-w-4xl">

            <div className="mb-5">

              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-blue-400">
                Analytics
              </p>

              <h2 className="mt-2 text-2xl font-bold text-white">
                Prediction Overview
              </h2>

            </div>


            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-5">

                <p className="text-sm text-slate-400">
                  Total Predictions
                </p>

                <p className="mt-2 text-3xl font-bold text-white">
                  {analytics.total}
                </p>

              </div>


              <div className="rounded-2xl border border-red-400/20 bg-red-400/5 p-5">

                <p className="text-sm text-red-300">
                  Fake Predictions
                </p>

                <p className="mt-2 text-3xl font-bold text-red-400">
                  {analytics.fake_count}
                </p>

                <p className="mt-1 text-xs text-slate-500">
                  {analytics.fake_percentage.toFixed(1)}%
                </p>

              </div>


              <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/5 p-5">

                <p className="text-sm text-emerald-300">
                  Real Predictions
                </p>

                <p className="mt-2 text-3xl font-bold text-emerald-400">
                  {analytics.real_count}
                </p>

                <p className="mt-1 text-xs text-slate-500">
                  {analytics.real_percentage.toFixed(1)}%
                </p>

              </div>


              <div className="rounded-2xl border border-blue-400/20 bg-blue-400/5 p-5">

                <p className="text-sm text-blue-300">
                  Model
                </p>

                <p className="mt-2 text-lg font-bold text-white">
                  Linear SVM
                </p>

                <p className="mt-1 text-xs text-slate-500">
                  TF-IDF based
                </p>

              </div>

            </div>

          </section>

        )}


        {/* RECENT PREDICTIONS HISTORY */}

        {history.length > 0 && (

          <section className="mx-auto mt-12 max-w-4xl">

            <div className="mb-5 flex items-center justify-between">

              <div>

                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-purple-400">
                  History
                </p>

                <h2 className="mt-2 text-2xl font-bold text-white">
                  Recent Predictions
                </h2>

              </div>


              <button
                onClick={loadAnalytics}
                disabled={historyLoading}
                className="rounded-xl border border-white/10 bg-white/5 p-3 text-slate-300 transition hover:bg-white/10 disabled:opacity-50"
                title="Refresh history"
              >

                <RefreshCw
                  size={18}
                  className={historyLoading ? "animate-spin" : ""}
                />

              </button>

            </div>


            <div className="space-y-3">

              {history.map((item) => (

                <div
                  key={item.id}
                  className="rounded-2xl border border-white/10 bg-white/[0.04] p-5"
                >

                  <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">

                    <div className="flex items-center gap-3">

                      <div
                        className={`rounded-xl p-2 ${
                          item.prediction === "FAKE"
                            ? "bg-red-400/10 text-red-400"
                            : "bg-emerald-400/10 text-emerald-400"
                        }`}
                      >

                        <History size={18} />

                      </div>

                      <div>

                        <p
                          className={`font-semibold ${
                            item.prediction === "FAKE"
                              ? "text-red-400"
                              : "text-emerald-400"
                          }`}
                        >

                          {item.prediction}

                        </p>

                        <p className="text-xs text-slate-500">
                          Prediction #{item.id}
                        </p>

                      </div>

                    </div>


                    <div className="text-left sm:text-right">

                      <p className="text-sm text-slate-300">
                        Score:{" "}
                        {item.decision_score.toFixed(4)}
                      </p>

                      <p className="mt-1 text-xs text-slate-500">
                        {new Date(
                          item.created_at
                        ).toLocaleString()}
                      </p>

                    </div>

                  </div>

                </div>

              ))}

            </div>


            <button
              onClick={handleClearHistory}
              className="mt-5 inline-flex items-center gap-2 rounded-xl border border-red-400/20 bg-red-400/5 px-4 py-2 text-sm font-semibold text-red-400 transition hover:bg-red-400/10"
            >

              <Trash2 size={16} />

              Clear History

            </button>

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
