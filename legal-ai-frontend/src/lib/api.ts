// lib/api.ts
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import qs from 'qs';
import { auth } from '@/lib/firebase';

// Interfaces
export interface SearchResponse {
  results: Array<{
    case_name: string;
    court: string;
    date_filed: string;
    excerpt: string;
    tech_keywords: string;
    similarity_score: number;
  }>;
  metadata: {
    total_cases: number;
    query_timestamp: string;
  };
}

export interface ClassificationResponse {
  predictions: string[];
  confidence_scores: Record<string, number>;
  detected_patterns: string[];
  enhanced_features: Record<string, number>;
  time_period: string;
  metadata: {
    timestamp: string;
    model_version: string;
    text_length: number;
  };
}

export interface PatentAnalysisResponse {
  analysis: {
    temporal_analysis: {
      total_applications: number;
      yearly_trends: Record<string, number>;
    };
    rejection_analysis: {
      total_rejections: number;
      most_common: [string, number];
    };
    claim_analysis: {
      most_rejected_claims: [number, number][];
    };
  };
  metadata: {
    timestamp: string;
    tech_center: string;
    analysis_version: string;
  };
}

// Create API instance
const createAPI = (): AxiosInstance => {
  const api = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000',
    // Use qs to serialize arrays as: keywords=tech&keywords=patent&keywords=software
    paramsSerializer: params => qs.stringify(params, { arrayFormat: 'repeat' }),
  });

  api.interceptors.request.use(async (config) => {
    try {
      const user = auth.currentUser;
      if (user) {
        const token = await user.getIdToken();
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    } catch (error) {
      console.error('Error getting auth token:', error);
      return config;
    }
  });

  api.interceptors.response.use(
    (response) => response,
    async (error) => {
      if (error.response?.status === 401) {
        window.location.href = '/login';
      }
      return Promise.reject(error);
    }
  );

  return api;
};



// Legal case endpoints
export const searchCases = async (query: string): Promise<SearchResponse> => {
  try {
    const response = await api.post('/search', { query });
    return response.data;
  } catch (error) {
    console.error('Error searching cases:', error);
    throw error;
  }
};

export const classifyCase = async (text: string): Promise<ClassificationResponse> => {
  try {
    const response = await api.post('/classify', { text });
    return response.data;
  } catch (error) {
    console.error('Error classifying case:', error);
    throw error;
  }
};

export const analyzeStrategy = async (text: string) => {
  try {
    const response = await api.post('/analyze_strategy', { text });
    return response.data;
  } catch (error) {
    console.error('Error analyzing strategy:', error);
    throw error;
  }
};

// Patent analysis endpoints
export const analyzeTechCenter = async (techCenter: string): Promise<PatentAnalysisResponse> => {
  try {
    const response = await api.post('/patent/analyze', { tech_center: techCenter });
    return response.data;
  } catch (error) {
    console.error('Error analyzing tech center:', error);
    throw error;
  }
};

export const getPatentHealth = async () => {
  try {
    const response = await api.get('/patent/health');
    return response.data;
  } catch (error) {
    console.error('Error getting patent health:', error);
    throw error;
  }
};

// Auth status check
export const checkAuthStatus = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    if (error.response?.status === 401) {
      return { authenticated: false };
    }
    throw error;
  }
};

export const getAvailableKeywords = async (): Promise<{ available_keywords: string[] }> => {
  try {
    const response = await api.get('/available_keywords');
    return response.data;
  } catch (error) {
    console.error('Error fetching available keywords:', error);
    throw error;
  }
};

// Analyze trends given a set of keywords
export const analyzeTrends = async (keywords: string[]): Promise<any> => {
  try {
    const response = await api.get('/analyze_trends', {
      // Axios automatically serializes arrays as repeated query parameters
      params: { keywords },
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing trends:', error);
    throw error;
  }
};

// Get keyword summary data
export const getKeywordSummary = async (): Promise<any> => {
  try {
    const response = await api.get('/keyword_summary');
    return response.data;
  } catch (error) {
    console.error('Error fetching keyword summary:', error);
    throw error;
  }
};

// Create API instance
const api = createAPI();

export default api;