import axios from "axios";
import { API_BASE_URL } from "../config";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

export async function predictNews(text) {
  const response = await api.post("/predict", { text });
  return response.data;
}

export async function explainNews(text) {
  const response = await api.post("/explain", { text });
  return response.data;
}

export async function getHealth() {
  const response = await api.get("/health");
  return response.data;
}

export async function checkBackend() {
  const response = await api.get("/health");
  return response.data;
}

export async function getModelInfo() {
  const response = await api.get("/model-info");
  return response.data;
}

export default api;
