// Production backend URL (GCE)
// Override locally by setting localStorage.setItem("API_BASE_URL", "http://localhost:8000")
const API_BASE_URL =
  localStorage.getItem("API_BASE_URL") ||
  "https://cmu-paper-reviewer.duckdns.org";
