/**
 * API Service for connecting to the Python backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Post {
  post_id: string;
  platform: string;
  text: string;
  image_path: string | null;
  urls: string[];
  hashtags: string[];
  likes: number;
  retweets: number;
  timestamp: string;
  account: {
    account_id: string;
    account_age_days: number;
    verified: boolean;
    historical_post_count: number;
    name?: string; // Account display name (e.g., "BBC")
    screen_name?: string; // Account handle (e.g., "BBC")
    description?: string; // Account description
    profile_image_url?: string; // Profile image URL
  };
}

export interface AnalysisResult {
  nlp_signals: {
    sentiment: string;
    emotion: string;
    clickbait: boolean;
    extracted_claim: string;
    text_embedding_id: string;
  };
  source_signals: {
    account_trust_score: number;
    source_reliability_score: number;
    behavioral_risk_flag: boolean;
  };
  image_signals: {
    ocr_text: string;
    image_tampered: boolean;
    ai_generated_probability: number;
  };
  misinformation_assessment: {
    content_credibility_score: number;
    risk_category: string;
  };
  final_decision: {
    final_credibility_score: number;
    agent_agreement_level: number;
    reasoning: string[];
  };
  user_facing_output: {
    credibility_score: number;
    warning_label: string;
    explanation: string[];
  };
}

export const api = {
  /**
   * Get the sample dataset
   */
  async getDataset(): Promise<Post[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/dataset`);
      const data = await response.json();
      return data.dataset || [];
    } catch (error) {
      console.error('Error fetching dataset:', error);
      return [];
    }
  },

  /**
   * Analyze a single post
   */
  async analyzePost(post: Post): Promise<AnalysisResult> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze-single`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(post),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error analyzing post:', error);
      throw error;
    }
  },

  /**
   * Analyze multiple posts
   */
  async analyzePosts(posts: Post[]): Promise<Record<string, AnalysisResult>> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ posts }),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      return data.results || {};
    } catch (error) {
      console.error('Error analyzing posts:', error);
      throw error;
    }
  },

  /**
   * Get saved results for a post
   */
  async getResults(postId: string): Promise<any> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/results/${postId}`);
      if (!response.ok) {
        throw new Error(`Results not found: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching results:', error);
      return null;
    }
  },

  /**
   * Check API health
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      return data.status === 'healthy';
    } catch (error) {
      console.error('API health check failed:', error);
      return false;
    }
  },

  /**
   * Get account details by account ID
   */
  async getAccountDetails(accountId: string): Promise<{ name?: string; screen_name?: string; description?: string } | null> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/account/${accountId}`);
      if (!response.ok) {
        return null;
      }
      const data = await response.json();
      return data.account || null;
    } catch (error) {
      console.error('Error fetching account details:', error);
      return null;
    }
  },
};
