
import { useEffect, useState } from "react";
import {
  Activity,
  ArrowLeftRight,
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  GitCompareArrows,
  LayoutDashboard,
  LoaderCircle,
  Package,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

type Product = {
  name: string;
  description: string;
  manufacturer: string;
  price: string;
};

type MatchResult = {
  success: boolean;
  model: string;
  match_probability: number;
  auto_match_threshold: number;
  auto_non_match_threshold: number;
  decision: "AUTO_MATCH" | "AUTO_NON_MATCH" | "HUMAN_REVIEW";
  features_generated: number;
  message: string;
};

const emptyProduct: Product = {
  name: "",
  description: "",
  manufacturer: "",
  price: "",
};

function App() {
  const [backendStatus, setBackendStatus] = useState<
    "Checking..." | "Connected" | "Disconnected"
  >("Checking...");

  const [productA, setProductA] = useState<Product>({
    ...emptyProduct,
  });

  const [productB, setProductB] = useState<Product>({
    ...emptyProduct,
  });

  const [result, setResult] = useState<MatchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function checkBackend() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);

      if (!response.ok) {
        throw new Error("Backend health check failed");
      }

      const data = await response.json();

      setBackendStatus(
        data.status === "healthy" && data.model_loaded
          ? "Connected"
          : "Disconnected"
      );
    } catch {
      setBackendStatus("Disconnected");
    }
  }

  useEffect(() => {
    checkBackend();

    const interval = window.setInterval(checkBackend, 15000);

    return () => window.clearInterval(interval);
  }, []);

  function updateProduct(
    side: "A" | "B",
    field: keyof Product,
    value: string
  ) {
    const setter = side === "A" ? setProductA : setProductB;

    setter((previous) => ({
      ...previous,
      [field]: value,
    }));

    setResult(null);
    setError("");
  }

  async function compareProducts(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/match`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          product_a: {
            ...productA,
            price: productA.price || null,
          },
          product_b: {
            ...productB,
            price: productB.price || null,
          },
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Unable to compare products."
        );
      }

      setResult(data as MatchResult);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  function resetComparison() {
    setProductA({ ...emptyProduct });
    setProductB({ ...emptyProduct });
    setResult(null);
    setError("");
  }

  const decisionDetails = {
    AUTO_MATCH: {
      title: "Automatic Match",
      description: "The model classified these products as a match.",
      className: "match",
      icon: <CheckCircle2 size={22} />,
    },
    AUTO_NON_MATCH: {
      title: "Not a Match",
      description: "The model classified these products as different.",
      className: "non-match",
      icon: <XCircle size={22} />,
    },
    HUMAN_REVIEW: {
      title: "Human Review Required",
      description: "The result needs a human decision.",
      className: "review",
      icon: <Clock3 size={22} />,
    },
  };

  const resultDetails = result
    ? decisionDetails[result.decision]
    : null;

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <GitCompareArrows size={23} />
          </div>
          <div>
            <h2>MatchWise</h2>
            <span>INTELLIGENT MATCHING</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav className="navigation">
          <a className="nav-item active" href="#dashboard">
            <LayoutDashboard size={19} />
            <span>Dashboard</span>
          </a>

          <a className="nav-item" href="#compare">
            <ArrowLeftRight size={19} />
            <span>Compare Products</span>
          </a>

          <a className="nav-item" href="#results">
            <BarChart3 size={19} />
            <span>Match Analysis</span>
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-info">
            <ShieldCheck size={20} />
            <div>
              <strong>Human in the loop</strong>
              <p>AI-assisted decisions</p>
            </div>
          </div>

          <div className="profile">
            <div className="avatar">MW</div>
            <div>
              <strong>MatchWise</strong>
              <p>ML Workspace</p>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-content" id="dashboard">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> Dashboard
          </div>

          <div className="backend-status">
            <span
              className={`status-dot ${
                backendStatus === "Connected"
                  ? "online"
                  : backendStatus === "Checking..."
                    ? "checking"
                    : "offline"
              }`}
            />
            <span>Backend: {backendStatus}</span>

            <button
              type="button"
              className="refresh-button"
              onClick={checkBackend}
              title="Refresh backend status"
            >
              <RefreshCw size={15} />
            </button>
          </div>
        </header>

        <section className="page-content">
          <div className="welcome-section">
            <div>
              <div className="eyebrow">
                <Sparkles size={15} />
                INTELLIGENT ENTITY RESOLUTION
              </div>

              <h1>Product Matching Dashboard</h1>

              <p>
                Identify duplicate products using machine learning
                and human-assisted decisions.
              </p>
            </div>

            <div className="date-label">
              <Activity size={17} />
              XGBoost powered
            </div>
          </div>

          <section className="stats-grid">
            <div className="stat-card">
              <div className="stat-icon blue">
                <Database size={21} />
              </div>
              <div>
                <span>ML Model</span>
                <h3>XGBoost</h3>
                <p>Active prediction model</p>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon teal">
                <Package size={21} />
              </div>
              <div>
                <span>Input Features</span>
                <h3>19</h3>
                <p>Features per comparison</p>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon purple">
                <ShieldCheck size={21} />
              </div>
              <div>
                <span>Auto-match threshold</span>
                <h3>99.75%</h3>
                <p>Validation-calibrated</p>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon orange">
                <Search size={21} />
              </div>
              <div>
                <span>Evaluation pairs</span>
                <h3>3,240</h3>
                <p>Held-out test dataset</p>
              </div>
            </div>
          </section>

          <section className="comparison-section" id="compare">
            <div className="section-heading">
              <div>
                <h2>Compare Products</h2>
                <p>
                  Enter product information to check whether two
                  records represent the same product.
                </p>
              </div>

              <button
                type="button"
                className="reset-button"
                onClick={resetComparison}
              >
                <RefreshCw size={15} />
                Reset
              </button>
            </div>

            <form onSubmit={compareProducts}>
              <div className="product-grid">
                <ProductForm
                  title="Product A"
                  subtitle="First product record"
                  product={productA}
                  onChange={(field, value) =>
                    updateProduct("A", field, value)
                  }
                  color="blue"
                />

                <div className="versus">
                  <div>
                    <ArrowLeftRight size={21} />
                  </div>
                  <span>VS</span>
                </div>

                <ProductForm
                  title="Product B"
                  subtitle="Second product record"
                  product={productB}
                  onChange={(field, value) =>
                    updateProduct("B", field, value)
                  }
                  color="teal"
                />
              </div>

              {error && (
                <div className="error-message">
                  <XCircle size={18} />
                  {error}
                </div>
              )}

              <div className="compare-actions">
                <p>
                  <ShieldCheck size={16} />
                  Powered by your trained XGBoost model
                </p>

                <button
                  className="compare-button"
                  type="submit"
                  disabled={loading || backendStatus !== "Connected"}
                >
                  {loading ? (
                    <>
                      <LoaderCircle className="spin" size={18} />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <GitCompareArrows size={18} />
                      Compare Products
                    </>
                  )}
                </button>
              </div>
            </form>
          </section>

          {result && resultDetails && (
            <section className="result-section" id="results">
              <div className="section-heading">
                <div>
                  <h2>Matching Analysis</h2>
                  <p>Live prediction returned by the MatchWise API.</p>
                </div>
                <span className="model-tag">XGBoost</span>
              </div>

              <div className={`result-card ${resultDetails.className}`}>
                <div className="result-header">
                  <div className="decision-icon">
                    {resultDetails.icon}
                  </div>

                  <div>
                    <h3>{resultDetails.title}</h3>
                    <p>{resultDetails.description}</p>
                  </div>
                </div>

                <div className="probability-area">
                  <div className="probability-heading">
                    <span>Match probability</span>
                    <strong>
                      {(result.match_probability * 100).toFixed(2)}%
                    </strong>
                  </div>

                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(0, result.match_probability * 100)
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="result-metrics">
                  <div>
                    <span>Auto-match threshold</span>
                    <strong>
                      {(result.auto_match_threshold * 100).toFixed(2)}%
                    </strong>
                  </div>

                  <div>
                    <span>Auto-non-match threshold</span>
                    <strong>
                      {(result.auto_non_match_threshold * 100).toFixed(2)}%
                    </strong>
                  </div>

                  <div>
                    <span>Features generated</span>
                    <strong>{result.features_generated}</strong>
                  </div>
                </div>
              </div>

              {result.decision === "HUMAN_REVIEW" && (
                <div className="review-note">
                  <Clock3 size={19} />
                  <div>
                    <strong>Manual verification recommended</strong>
                    <p>
                      This result is in the review range. A human
                      reviewer should make the final decision.
                    </p>
                  </div>
                </div>
              )}
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

type ProductFormProps = {
  title: string;
  subtitle: string;
  product: Product;
  color: "blue" | "teal";
  onChange: (field: keyof Product, value: string) => void;
};

function ProductForm({
  title,
  subtitle,
  product,
  color,
  onChange,
}: ProductFormProps) {
  return (
    <div className={`product-card ${color}`}>
      <div className="product-card-heading">
        <div className="product-symbol">
          <Package size={21} />
        </div>

        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
      </div>

      <div className="form-fields">
        <label>
          Product name <span>*</span>
          <input
            type="text"
            placeholder="e.g. Samsung Galaxy S24 128GB"
            value={product.name}
            onChange={(event) =>
              onChange("name", event.target.value)
            }
            required
          />
        </label>

        <label>
          Description
          <textarea
            placeholder="Enter product description..."
            rows={3}
            value={product.description}
            onChange={(event) =>
              onChange("description", event.target.value)
            }
          />
        </label>

        <label>
          Manufacturer
          <input
            type="text"
            placeholder="e.g. Samsung"
            value={product.manufacturer}
            onChange={(event) =>
              onChange("manufacturer", event.target.value)
            }
          />
        </label>

        <label>
          Price
          <input
            type="number"
            min="0"
            step="any"
            placeholder="e.g. 69999"
            value={product.price}
            onChange={(event) =>
              onChange("price", event.target.value)
            }
          />
        </label>
      </div>
    </div>
  );
}

export default App;
