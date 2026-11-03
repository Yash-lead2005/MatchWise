
import { useCallback, useEffect, useState } from "react";
import {
  Activity, ArrowLeftRight, CheckCircle2, Clock3,
  Database, GitCompareArrows, LayoutDashboard, LoaderCircle,
  Package, RefreshCw, Search, ShieldCheck, Sparkles, XCircle,
  History, ClipboardCheck, AlertCircle
} from "lucide-react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend
} from "recharts";

import "./App.css";

const API = "http://127.0.0.1:8000";

type Product = {
  name: string;
  description: string;
  manufacturer: string;
  price: string;
};

type MatchResult = {
  success: boolean;
  record_id: number;
  model: string;
  match_probability: number;
  auto_match_threshold: number;
  auto_non_match_threshold: number;
  decision: "AUTO_MATCH" | "AUTO_NON_MATCH" | "HUMAN_REVIEW";
  review_status: string;
  features_generated: number;
  message: string;
};

type Analytics = {
  total_comparisons: number;
  auto_matches: number;
  auto_non_matches: number;
  pending_reviews: number;
  completed_reviews: number;
  human_confirmed_matches: number;
  human_confirmed_non_matches: number;
  average_match_probability: number;
};

type Review = {
  id: number;
  product_a: Product;
  product_b: Product;
  match_probability: number;
  model_decision: string;
  review_status: string;
  created_at: string;
};

type HistoryRecord = Review & {
  human_decision: string | null;
  final_decision: string;
  reviewed_at: string | null;
};

type Page = "dashboard" | "compare" | "reviews" | "history";

const emptyProduct: Product = {
  name: "",
  description: "",
  manufacturer: "",
  price: ""
};

const emptyAnalytics: Analytics = {
  total_comparisons: 0,
  auto_matches: 0,
  auto_non_matches: 0,
  pending_reviews: 0,
  completed_reviews: 0,
  human_confirmed_matches: 0,
  human_confirmed_non_matches: 0,
  average_match_probability: 0
};

function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [analytics, setAnalytics] = useState<Analytics>(emptyAnalytics);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [productA, setProductA] = useState<Product>({ ...emptyProduct });
  const [productB, setProductB] = useState<Product>({ ...emptyProduct });
  const [result, setResult] = useState<MatchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [pageLoading, setPageLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [search, setSearch] = useState("");
  const [actionId, setActionId] = useState<number | null>(null);

  const request = async (path: string, options?: RequestInit) => {
    const response = await fetch(`${API}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers
      }
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : "The request could not be completed."
      );
    }

    return data;
  };

  const checkBackend = useCallback(async () => {
    try {
      const data = await request("/health");
      setBackendStatus(
        data.status === "healthy" && data.model_loaded
          ? "Connected"
          : "Disconnected"
      );
    } catch {
      setBackendStatus("Disconnected");
    }
  }, []);

  const loadAnalytics = useCallback(async () => {
    try {
      const data = await request("/analytics");
      setAnalytics({
        ...emptyAnalytics,
        ...data
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analytics could not load.");
    }
  }, []);

  const loadReviews = useCallback(async () => {
    try {
      const data = await request("/reviews/pending");
      setReviews(data.reviews || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reviews could not load.");
    }
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      const data = await request("/history?limit=100&offset=0");
      setHistory(data.history || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "History could not load.");
    }
  }, []);

  const refreshAll = useCallback(async () => {
    setPageLoading(true);
    setError("");
    await Promise.all([
      checkBackend(),
      loadAnalytics(),
      loadReviews(),
      loadHistory()
    ]);
    setPageLoading(false);
  }, [checkBackend, loadAnalytics, loadReviews, loadHistory]);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  function updateProduct(
    side: "A" | "B",
    field: keyof Product,
    value: string
  ) {
    const setter = side === "A" ? setProductA : setProductB;
    setter(previous => ({ ...previous, [field]: value }));
    setResult(null);
    setError("");
    setNotice("");
  }

  async function compareProducts(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setNotice("");
    setResult(null);

    try {
      const data = await request("/match", {
        method: "POST",
        body: JSON.stringify({
          product_a: { ...productA, price: productA.price || null },
          product_b: { ...productB, price: productB.price || null }
        })
      });

      setResult(data as MatchResult);
      setNotice(`Comparison saved successfully. Record ID: ${data.record_id}`);
      await Promise.all([loadAnalytics(), loadReviews(), loadHistory()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed.");
    } finally {
      setLoading(false);
    }
  }

  async function submitDecision(id: number, decision: "MATCH" | "NON_MATCH") {
    setActionId(id);
    setError("");
    setNotice("");

    try {
      await request(`/reviews/${id}/decision`, {
        method: "POST",
        body: JSON.stringify({ decision })
      });

      setNotice(`Review #${id} successfully marked as ${decision}.`);
      await Promise.all([loadAnalytics(), loadReviews(), loadHistory()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Review action failed.");
    } finally {
      setActionId(null);
    }
  }

  function resetComparison() {
    setProductA({ ...emptyProduct });
    setProductB({ ...emptyProduct });
    setResult(null);
    setError("");
    setNotice("");
  }

  const filteredHistory = history.filter(record => {
    const query = search.toLowerCase();
    return (
      record.product_a.name.toLowerCase().includes(query) ||
      record.product_b.name.toLowerCase().includes(query) ||
      record.final_decision.toLowerCase().includes(query) ||
      String(record.id).includes(query)
    );
  });

  const decisionDetails = {
    AUTO_MATCH: {
      title: "Automatic Match",
      description: "The model classified these products as a match.",
      className: "match",
      icon: <CheckCircle2 size={22} />
    },
    AUTO_NON_MATCH: {
      title: "Not a Match",
      description: "The model classified these products as different.",
      className: "non-match",
      icon: <XCircle size={22} />
    },
    HUMAN_REVIEW: {
      title: "Human Review Required",
      description: "A human reviewer should make the final decision.",
      className: "review",
      icon: <Clock3 size={22} />
    }
  };

  const resultDetails = result ? decisionDetails[result.decision] : null;

  const pageTitles: Record<Page, string> = {
    dashboard: "Dashboard",
    compare: "Compare Products",
    reviews: "Human Review Queue",
    history: "Matching History"
  };

  const chartData = [
    { name: "Auto Match", value: analytics.auto_matches },
    { name: "Auto Non-Match", value: analytics.auto_non_matches },
    { name: "Human Reviewed", value: analytics.completed_reviews },
    { name: "Pending", value: analytics.pending_reviews }
  ];

  const COLORS = ["#159b78", "#e36b70", "#6888d8", "#e5a343"];

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><GitCompareArrows size={23} /></div>
          <div>
            <h2>MatchWise</h2>
            <span>INTELLIGENT MATCHING</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav className="navigation">
          <button className={`nav-item ${page === "dashboard" ? "active" : ""}`} onClick={() => setPage("dashboard")}>
            <LayoutDashboard size={19} /><span>Dashboard</span>
          </button>
          <button className={`nav-item ${page === "compare" ? "active" : ""}`} onClick={() => setPage("compare")}>
            <ArrowLeftRight size={19} /><span>Compare Products</span>
          </button>
          <button className={`nav-item ${page === "reviews" ? "active" : ""}`} onClick={() => setPage("reviews")}>
            <ClipboardCheck size={19} /><span>Human Review</span>
            {analytics.pending_reviews > 0 && <b className="nav-count">{analytics.pending_reviews}</b>}
          </button>
          <button className={`nav-item ${page === "history" ? "active" : ""}`} onClick={() => setPage("history")}>
            <History size={19} /><span>Matching History</span>
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-info">
            <ShieldCheck size={20} />
            <div><strong>Human in the loop</strong><p>AI-assisted decisions</p></div>
          </div>
          <div className="profile">
            <div className="avatar">MW</div>
            <div><strong>MatchWise</strong><p>ML Workspace</p></div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> {pageTitles[page]}</div>
          <div className="backend-status">
            <span className={`status-dot ${backendStatus === "Connected" ? "online" : backendStatus === "Checking..." ? "checking" : "offline"}`} />
            <span>Backend: {backendStatus}</span>
            <button className="refresh-button" onClick={refreshAll} title="Refresh data">
              <RefreshCw size={15} />
            </button>
          </div>
        </header>

        <section className="page-content">
          <div className="welcome-section">
            <div>
              <div className="eyebrow"><Sparkles size={15} /> INTELLIGENT ENTITY RESOLUTION</div>
              <h1>{pageTitles[page]}</h1>
              <p>Identify duplicate products using machine learning and human-assisted decisions.</p>
            </div>
            <div className="date-label"><Activity size={17} /> XGBoost powered</div>
          </div>

          {error && <div className="error-message"><AlertCircle size={18} />{error}</div>}
          {notice && <div className="success-message"><CheckCircle2 size={18} />{notice}</div>}

          {page === "dashboard" && (
            <>
              <section className="stats-grid">
                <StatCard icon={<Database size={21}/>} color="blue" label="Total comparisons" value={analytics.total_comparisons} note="Saved in database"/>
                <StatCard icon={<CheckCircle2 size={21}/>} color="teal" label="Auto matches" value={analytics.auto_matches} note="Model decisions"/>
                <StatCard icon={<XCircle size={21}/>} color="purple" label="Auto non-matches" value={analytics.auto_non_matches} note="Model decisions"/>
                <StatCard icon={<Clock3 size={21}/>} color="orange" label="Pending reviews" value={analytics.pending_reviews} note="Awaiting human decision"/>
              </section>

              <div className="analytics-grid">
                <section className="panel">
                  <div className="section-heading">
                    <div><h2>Decision distribution</h2><p>Comparison outcomes from your database.</p></div>
                  </div>
                  {analytics.total_comparisons === 0 ? <EmptyState text="No comparison data yet. Start by comparing two products."/> :
                    <div className="chart-box">
                      <ResponsiveContainer width="100%" height={260}>
                        <PieChart>
                          <Pie data={chartData.filter(item => item.value > 0)} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={85} label>
                            {chartData.filter(item => item.value > 0).map((item, index) => <Cell key={item.name} fill={COLORS[index]}/>)}
                          </Pie>
                          <Tooltip/><Legend/>
                        </PieChart>
                      </ResponsiveContainer>
                    </div>}
                </section>

                <section className="panel">
                  <div className="section-heading"><div><h2>Human review summary</h2><p>Reviewer decisions recorded.</p></div></div>
                  <div className="review-summary">
                    <div><span>Completed reviews</span><strong>{analytics.completed_reviews}</strong></div>
                    <div><span>Confirmed matches</span><strong>{analytics.human_confirmed_matches}</strong></div>
                    <div><span>Confirmed non-matches</span><strong>{analytics.human_confirmed_non_matches}</strong></div>
                    <div><span>Average match probability</span><strong>{(analytics.average_match_probability * 100).toFixed(2)}%</strong></div>
                  </div>
                  <button className="secondary-button" onClick={() => setPage("reviews")}>Open review queue <ArrowLeftRight size={15}/></button>
                </section>
              </div>

              <section className="panel">
                <div className="section-heading">
                  <div><h2>Model overview</h2><p>Current matching engine configuration.</p></div>
                  <span className="model-tag">ACTIVE · XGBoost</span>
                </div>
                <div className="overview-grid">
                  <div><span>Input features</span><strong>19</strong></div>
                  <div><span>Auto-match threshold</span><strong>99.75%</strong></div>
                  <div><span>Auto non-match threshold</span><strong>91.33%</strong></div>
                  <div><span>Completed human reviews</span><strong>{analytics.completed_reviews}</strong></div>
                </div>
              </section>
            </>
          )}

          {page === "compare" && (
            <>
              <section className="comparison-section" id="compare">
                <div className="section-heading">
                  <div><h2>Compare Products</h2><p>Enter two product records to check whether they represent the same product.</p></div>
                  <button className="reset-button" onClick={resetComparison}><RefreshCw size={15}/> Reset</button>
                </div>
                <form onSubmit={compareProducts}>
                  <div className="product-grid">
                    <ProductForm title="Product A" subtitle="First product record" product={productA} onChange={(field, value) => updateProduct("A", field, value)} color="blue"/>
                    <div className="versus"><div><ArrowLeftRight size={21}/></div><span>VS</span></div>
                    <ProductForm title="Product B" subtitle="Second product record" product={productB} onChange={(field, value) => updateProduct("B", field, value)} color="teal"/>
                  </div>
                  <div className="compare-actions">
                    <p><ShieldCheck size={16}/> Powered by your trained XGBoost model</p>
                    <button className="compare-button" type="submit" disabled={loading || backendStatus !== "Connected"}>
                      {loading ? <><LoaderCircle className="spin" size={18}/> Analyzing...</> : <><GitCompareArrows size={18}/> Compare Products</>}
                    </button>
                  </div>
                </form>
              </section>

              {result && resultDetails && (
                <section className="result-section">
                  <div className="section-heading"><div><h2>Matching Analysis</h2><p>Live prediction returned by MatchWise API. Record #{result.record_id} saved.</p></div><span className="model-tag">XGBoost</span></div>
                  <div className={`result-card ${resultDetails.className}`}>
                    <div className="result-header"><div className="decision-icon">{resultDetails.icon}</div><div><h3>{resultDetails.title}</h3><p>{resultDetails.description}</p></div></div>
                    <div className="probability-area">
                      <div className="probability-heading"><span>Match probability</span><strong>{(result.match_probability * 100).toFixed(2)}%</strong></div>
                      <div className="progress-track"><div className="progress-fill" style={{width: `${Math.min(100, Math.max(0, result.match_probability * 100))}%`}}/></div>
                    </div>
                    <div className="result-metrics">
                      <div><span>Auto-match threshold</span><strong>{(result.auto_match_threshold * 100).toFixed(2)}%</strong></div>
                      <div><span>Auto-non-match threshold</span><strong>{(result.auto_non_match_threshold * 100).toFixed(2)}%</strong></div>
                      <div><span>Features generated</span><strong>{result.features_generated}</strong></div>
                    </div>
                  </div>
                  {result.decision === "HUMAN_REVIEW" && <div className="review-note"><Clock3 size={19}/><div><strong>Manual verification recommended</strong><p>This comparison has been added to the Human Review queue.</p><button className="text-button" onClick={() => setPage("reviews")}>Open review queue</button></div></div>}
                </section>
              )}
            </>
          )}

          {page === "reviews" && (
            <section className="panel">
              <div className="section-heading">
                <div><h2>Pending human reviews</h2><p>Review uncertain product pairs and record the final decision.</p></div>
                <button className="reset-button" onClick={loadReviews}><RefreshCw size={15}/> Refresh</button>
              </div>
              {pageLoading && <p className="muted">Loading reviews...</p>}
              {!pageLoading && reviews.length === 0 && <EmptyState text="No pending reviews. New uncertain comparisons will appear here."/>}
              <div className="review-list">
                {reviews.map(review => (
                  <div className="review-item" key={review.id}>
                    <div className="review-item-top"><strong>Review #{review.id}</strong><span className="status-pill pending">PENDING</span></div>
                    <div className="review-products">
                      <div><span>PRODUCT A</span><strong>{review.product_a.name}</strong><p>{review.product_a.description || "No description provided"}</p><small>Manufacturer: {review.product_a.manufacturer || "Not provided"} · Price: {review.product_a.price || "N/A"}</small></div>
                      <div className="review-vs">VS</div>
                      <div><span>PRODUCT B</span><strong>{review.product_b.name}</strong><p>{review.product_b.description || "No description provided"}</p><small>Manufacturer: {review.product_b.manufacturer || "Not provided"} · Price: {review.product_b.price || "N/A"}</small></div>
                    </div>
                    <div className="review-item-bottom">
                      <span>Model probability: <strong>{(review.match_probability * 100).toFixed(2)}%</strong></span>
                      <div className="review-actions">
                        <button className="approve-button" disabled={actionId === review.id} onClick={() => submitDecision(review.id, "MATCH")}>{actionId === review.id ? <LoaderCircle className="spin" size={15}/> : <CheckCircle2 size={15}/>} Approve match</button>
                        <button className="reject-button" disabled={actionId === review.id} onClick={() => submitDecision(review.id, "NON_MATCH")}><XCircle size={15}/> Reject match</button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {page === "history" && (
            <section className="panel">
              <div className="section-heading">
                <div><h2>All comparisons</h2><p>{history.length} records retrieved from the MatchWise database.</p></div>
                <button className="reset-button" onClick={loadHistory}><RefreshCw size={15}/> Refresh</button>
              </div>
              <div className="history-search"><Search size={17}/><input placeholder="Search product name, decision or record ID..." value={search} onChange={event => setSearch(event.target.value)}/></div>
              <div className="table-wrap">
                <table className="history-table">
                  <thead><tr><th>ID</th><th>Product A</th><th>Product B</th><th>Probability</th><th>Model decision</th><th>Final decision</th><th>Status</th></tr></thead>
                  <tbody>
                    {filteredHistory.map(record => (
                      <tr key={record.id}>
                        <td>#{record.id}</td>
                        <td>{record.product_a.name}</td>
                        <td>{record.product_b.name}</td>
                        <td>{(record.match_probability * 100).toFixed(2)}%</td>
                        <td>{record.model_decision}</td>
                        <td><strong>{record.final_decision}</strong></td>
                        <td><span className={`status-pill ${record.review_status === "PENDING" ? "pending" : "reviewed"}`}>{record.review_status}</span></td>
                      </tr>
                    ))}
                    {filteredHistory.length === 0 && <tr><td colSpan={7}><EmptyState text="No matching history records found."/></td></tr>}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          <footer className="footer">
            <span>© 2026 MatchWise · Intelligent Entity Resolution</span>
            <span>Built with React, FastAPI & XGBoost</span>
          </footer>
        </section>
      </main>
    </div>
  );
}

function StatCard({ icon, color, label, value, note }: {
  icon: React.ReactNode;
  color: string;
  label: string;
  value: number;
  note: string;
}) {
  return <div className="stat-card"><div className={`stat-icon ${color}`}>{icon}</div><div><span>{label}</span><h3>{value}</h3><p>{note}</p></div></div>;
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty-state"><Database size={25}/><p>{text}</p></div>;
}

type ProductFormProps = {
  title: string;
  subtitle: string;
  product: Product;
  color: "blue" | "teal";
  onChange: (field: keyof Product, value: string) => void;
};

function ProductForm({ title, subtitle, product, color, onChange }: ProductFormProps) {
  return (
    <div className={`product-card ${color}`}>
      <div className="product-card-heading"><div className="product-symbol"><Package size={21}/></div><div><h3>{title}</h3><p>{subtitle}</p></div></div>
      <div className="form-fields">
        <label>Product name <span>*</span><input type="text" placeholder="e.g. Samsung Galaxy S24 128GB" value={product.name} onChange={event => onChange("name", event.target.value)} required/></label>
        <label>Description<textarea placeholder="Enter product description..." rows={3} value={product.description} onChange={event => onChange("description", event.target.value)}/></label>
        <label>Manufacturer<input type="text" placeholder="e.g. Samsung" value={product.manufacturer} onChange={event => onChange("manufacturer", event.target.value)}/></label>
        <label>Price<input type="number" min="0" step="any" placeholder="e.g. 69999" value={product.price} onChange={event => onChange("price", event.target.value)}/></label>
      </div>
    </div>
  );
}

export default App;
