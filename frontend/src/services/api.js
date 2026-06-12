import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: { 'Content-Type': 'application/json' },
});

export const uploadPDF = async (files) => {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));

  const response = await api.post('/api/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const queryDocuments = async (payload) => {
  const response = await api.post('/api/query', payload);
  return response.data;
};

export const runEvaluation = async () => {
  const response = await api.post('/api/evaluate');
  return response.data;
};

export const getMetrics = async () => {
  const response = await api.get('/api/evaluate/metrics');
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};
