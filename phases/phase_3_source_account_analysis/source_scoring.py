"""
Phase 3: Source Account Analysis Pipeline
Assesses credibility based on account behavior and source reliability.

This module implements:
- Account trust scoring (rule-based)
- Domain reliability scoring
- Behavioral heuristics

All outputs are stored keyed by post_id and match the source_signals schema.
Uses rule-based scoring only (no machine learning).
"""

import json
import os
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import re


class SourceAccountAnalyzer:
    """
    Rule-based analyzer for account credibility and source reliability.
    Follows phase contracts strictly and uses only rule-based heuristics.
    """
    
    def __init__(self, known_domain_list: Optional[List[str]] = None, 
                 blacklisted_sources: Optional[List[str]] = None):
        """
        Initialize the source account analyzer.
        
        Args:
            known_domain_list: List of trusted/known domains (optional)
            blacklisted_sources: List of blacklisted domains (optional)
        """
        self.known_domain_list = known_domain_list or []
        self.blacklisted_sources = blacklisted_sources or []
        
        # Normalize domain lists (lowercase, remove www, etc.)
        self.known_domain_list = [self._normalize_domain(d) for d in self.known_domain_list]
        self.blacklisted_sources = [self._normalize_domain(d) for d in self.blacklisted_sources]
    
    def _normalize_domain(self, domain: str) -> str:
        """
        Normalize domain string for comparison.
        
        Args:
            domain: Domain string (can be URL or just domain)
            
        Returns:
            Normalized domain string
        """
        # If it's a URL, extract domain
        if '://' in domain:
            parsed = urlparse(domain)
            domain = parsed.netloc or parsed.path
        
        # Remove www. prefix and convert to lowercase
        domain = domain.lower().replace('www.', '').strip()
        return domain
    
    def _extract_domains_from_urls(self, urls: List[str]) -> List[str]:
        """
        Extract domains from a list of URLs.
        
        Args:
            urls: List of URL strings
            
        Returns:
            List of normalized domain strings
        """
        domains = []
        for url in urls:
            if url:
                domain = self._normalize_domain(url)
                if domain:
                    domains.append(domain)
        return domains
    
    def compute_account_trust_score(self, account_age_days: int, 
                                   verified: bool, 
                                   historical_post_count: int,
                                   name: Optional[str] = None,
                                   screen_name: Optional[str] = None,
                                   description: Optional[str] = None,
                                   followers_count: Optional[int] = None) -> float:
        """
        Compute account trust score based on account characteristics.
        Uses rule-based heuristics, normalized to 0-1 range.
        
        Args:
            account_age_days: Age of account in days
            verified: Whether account is verified
            historical_post_count: Number of historical posts
            name: Account display name (optional)
            screen_name: Account handle (optional)
            description: Account description (optional)
            followers_count: Number of followers (optional)
            
        Returns:
            Account trust score (float between 0 and 1)
        """
        score = 0.0
        
        # Factor 1: Account age (0-0.4 points)
        # Older accounts are more trustworthy
        # Max score at 365+ days
        if account_age_days >= 365:
            age_score = 0.4
        elif account_age_days >= 180:
            age_score = 0.3
        elif account_age_days >= 90:
            age_score = 0.2
        elif account_age_days >= 30:
            age_score = 0.1
        else:
            # Very new accounts get minimal score
            age_score = 0.05 * (account_age_days / 30.0) if account_age_days > 0 else 0.0
        
        score += age_score
        
        # Factor 2: Verification status (0-0.3 points)
        if verified:
            score += 0.3
        
        # Factor 3: Historical post count (0-0.2 points)
        # Accounts with more posts are generally more established
        # But too many posts in short time might be spam
        if historical_post_count > 0:
            if historical_post_count >= 1000:
                # Established account
                post_score = 0.2
            elif historical_post_count >= 500:
                post_score = 0.15
            elif historical_post_count >= 100:
                post_score = 0.1
            elif historical_post_count >= 50:
                post_score = 0.08
            elif historical_post_count >= 10:
                post_score = 0.05
            else:
                post_score = 0.03
        else:
            post_score = 0.0
        
        score += post_score
        
        # Factor 4: Known trusted sources (0-0.1 points)
        # Check if account name/screen_name matches known trusted sources
        if name or screen_name:
            trusted_sources = [
                'bbc', 'reuters', 'ap news', 'associated press', 'the guardian',
                'new york times', 'washington post', 'wall street journal',
                'cnn', 'npr', 'pbs', 'propublica'
            ]
            account_text = f"{name or ''} {screen_name or ''}".lower()
            if any(trusted in account_text for trusted in trusted_sources):
                score += 0.1
        
        # Factor 5: Followers count (0-0.05 points)
        # High follower count indicates established account
        if followers_count:
            if followers_count >= 1000000:
                score += 0.05
            elif followers_count >= 100000:
                score += 0.03
            elif followers_count >= 10000:
                score += 0.02
            elif followers_count >= 1000:
                score += 0.01
        
        # Factor 6: Description quality (0-0.05 points)
        # Professional descriptions indicate legitimate accounts
        if description:
            description_lower = description.lower()
            professional_indicators = [
                'news', 'media', 'broadcaster', 'journalism', 'official',
                'organization', 'institution', 'public service'
            ]
            if any(indicator in description_lower for indicator in professional_indicators):
                score += 0.05
        
        # Normalize to 0-1 range (already should be, but ensure)
        return min(1.0, max(0.0, score))
    
    def evaluate_domain_reliability(self, urls: List[str]) -> float:
        """
        Evaluate domain reliability score based on URLs in the post.
        Uses rule-based heuristics, normalized to 0-1 range.
        
        Args:
            urls: List of URLs in the post
            
        Returns:
            Domain reliability score (float between 0 and 1)
        """
        if not urls:
            # No URLs means neutral reliability (0.5)
            return 0.5
        
        domains = self._extract_domains_from_urls(urls)
        if not domains:
            return 0.5
        
        scores = []
        
        for domain in domains:
            domain_score = 0.5  # Default neutral score
            
            # Check if domain is blacklisted
            if domain in self.blacklisted_sources:
                domain_score = 0.0
            # Check if domain is in known trusted list
            elif domain in self.known_domain_list:
                domain_score = 1.0
            else:
                # Apply heuristics based on domain characteristics
                
                # Heuristic 1: Common trusted TLDs
                trusted_tlds = ['.edu', '.gov', '.org', '.mil']
                if any(domain.endswith(tld) for tld in trusted_tlds):
                    domain_score = 0.7
                
                # Heuristic 2: Suspicious patterns
                suspicious_patterns = [
                    r'bit\.ly', r'tinyurl', r'short\.link', r'goo\.gl',
                    r't\.co', r'ow\.ly', r'is\.gd', r'clck\.ru'
                ]
                if any(re.search(pattern, domain, re.IGNORECASE) for pattern in suspicious_patterns):
                    domain_score = 0.3
                
                # Heuristic 3: News/media domains (generally more reliable)
                news_indicators = ['news', 'media', 'press', 'reuters', 'ap', 'bbc', 'cnn']
                if any(indicator in domain for indicator in news_indicators):
                    domain_score = min(0.8, domain_score + 0.2)
                
                # Heuristic 4: Social media domains (moderate reliability)
                social_domains = ['twitter.com', 'facebook.com', 'instagram.com', 
                                 'linkedin.com', 'youtube.com', 'tiktok.com']
                if any(social in domain for social in social_domains):
                    domain_score = 0.6
            
            scores.append(domain_score)
        
        # Average scores if multiple domains
        avg_score = sum(scores) / len(scores) if scores else 0.5
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, avg_score))
    
    def apply_behavioral_heuristics(self, account_age_days: int,
                                    verified: bool,
                                    historical_post_count: int,
                                    urls: List[str]) -> bool:
        """
        Apply behavioral heuristics to detect risky account behavior.
        
        Args:
            account_age_days: Age of account in days
            verified: Whether account is verified
            historical_post_count: Number of historical posts
            urls: List of URLs in the post
            
        Returns:
            True if behavioral risk is detected, False otherwise
        """
        risk_flags = []
        
        # Risk 1: Very new account with many posts (potential bot/spam)
        if account_age_days < 30 and historical_post_count > 100:
            risk_flags.append("new_account_high_activity")
        
        # Risk 2: Account with no posts (suspicious)
        if historical_post_count == 0:
            risk_flags.append("no_historical_posts")
        
        # Risk 3: Very new account (less than 7 days)
        if account_age_days < 7:
            risk_flags.append("very_new_account")
        
        # Risk 4: Unverified account with suspicious URL patterns
        if not verified and urls:
            domains = self._extract_domains_from_urls(urls)
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']  # Free domains often used for spam
            if any(domain.endswith(tld) for domain in domains for tld in suspicious_tlds):
                risk_flags.append("suspicious_domain_patterns")
        
        # Risk 5: Multiple URL shorteners (potential spam/phishing)
        if urls:
            shortener_patterns = [
                r'bit\.ly', r'tinyurl', r'short\.link', r'goo\.gl',
                r'ow\.ly', r'is\.gd', r'clck\.ru', r't\.co'
            ]
            shortener_count = sum(
                1 for url in urls 
                for pattern in shortener_patterns 
                if re.search(pattern, url, re.IGNORECASE)
            )
            if shortener_count >= 2:
                risk_flags.append("multiple_url_shorteners")
        
        # Risk 6: Account age vs post count ratio (potential bot)
        if account_age_days > 0:
            posts_per_day = historical_post_count / account_age_days
            if posts_per_day > 10:  # More than 10 posts per day on average
                risk_flags.append("unusually_high_posting_rate")
        
        # Return True if any risk flags are detected
        return len(risk_flags) > 0
    
    def process_post(self, post_id: str,
                    account_age_days: int,
                    verified: bool,
                    historical_post_count: int,
                    urls: List[str] = None,
                    name: Optional[str] = None,
                    screen_name: Optional[str] = None,
                    description: Optional[str] = None,
                    followers_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a single post and compute all source signals.
        
        Args:
            post_id: Unique identifier for the post
            account_age_days: Age of account in days
            verified: Whether account is verified
            historical_post_count: Number of historical posts
            urls: List of URLs in the post (optional)
            name: Account display name (optional)
            screen_name: Account handle (optional)
            description: Account description (optional)
            followers_count: Number of followers (optional)
            
        Returns:
            Dictionary matching source_signals schema
        """
        if urls is None:
            urls = []
        
        # Compute all scores
        account_trust_score = self.compute_account_trust_score(
            account_age_days, verified, historical_post_count,
            name, screen_name, description, followers_count
        )
        
        source_reliability_score = self.evaluate_domain_reliability(urls)
        
        behavioral_risk_flag = self.apply_behavioral_heuristics(
            account_age_days, verified, historical_post_count, urls
        )
        
        # Return in exact schema format
        return {
            "account_trust_score": round(account_trust_score, 4),
            "source_reliability_score": round(source_reliability_score, 4),
            "behavioral_risk_flag": behavioral_risk_flag
        }
    
    def process_batch(self, posts: List[Dict[str, Any]], 
                     output_file: str = "source_outputs.json"):
        """
        Process multiple posts and save results keyed by post_id.
        
        Args:
            posts: List of post dictionaries with account and URL information
            output_file: Path to output JSON file
        """
        results = {}
        
        for i, post in enumerate(posts, 1):
            post_id = post.get('post_id', f'post_{i}')
            
            # Extract account information
            account = post.get('account', {})
            account_age_days = account.get('account_age_days', 0)
            verified = account.get('verified', False)
            historical_post_count = account.get('historical_post_count', 0)
            name = account.get('name')
            screen_name = account.get('screen_name')
            description = account.get('description')
            followers_count = account.get('followers_count')
            
            # Extract URLs
            urls = post.get('urls', [])
            
            print(f"Processing post {i}/{len(posts)}: {post_id}")
            
            try:
                source_signals = self.process_post(
                    post_id, account_age_days, verified, 
                    historical_post_count, urls,
                    name, screen_name, description, followers_count
                )
                results[post_id] = source_signals
            except Exception as e:
                print(f"Error processing post {post_id}: {e}")
                # Return default signals on error
                results[post_id] = {
                    "account_trust_score": 0.5,
                    "source_reliability_score": 0.5,
                    "behavioral_risk_flag": False
                }
        
        # Save results to JSON file
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Processed {len(results)} posts successfully")
        
        return results


def main():
    """
    Main entry point for the source account analysis pipeline.
    Can be used for testing or standalone execution.
    """
    import sys
    
    # Initialize analyzer with optional domain lists
    known_domains = ['reuters.com', 'bbc.com', 'ap.org', 'nytimes.com']
    blacklisted = ['example-spam.com', 'fake-news.net']
    
    analyzer = SourceAccountAnalyzer(
        known_domain_list=known_domains,
        blacklisted_sources=blacklisted
    )
    
    # Example usage with sample data
    if len(sys.argv) > 1:
        # Load input from JSON file
        input_file = sys.argv[1]
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Handle both single post and list of posts
        if isinstance(input_data, dict):
            if 'post_id' in input_data:
                posts = [input_data]
            else:
                # Assume it's a dict of posts keyed by post_id
                posts = [{'post_id': k, **v} for k, v in input_data.items()]
        else:
            posts = input_data
        
        output_file = sys.argv[2] if len(sys.argv) > 2 else "source_outputs.json"
        analyzer.process_batch(posts, output_file)
    else:
        # Run with sample data
        sample_posts = [
            {
                "post_id": "sample_1",
                "account": {
                    "account_age_days": 730,
                    "verified": True,
                    "historical_post_count": 1500
                },
                "urls": ["https://www.reuters.com/article/example"]
            },
            {
                "post_id": "sample_2",
                "account": {
                    "account_age_days": 5,
                    "verified": False,
                    "historical_post_count": 200
                },
                "urls": ["https://bit.ly/short-link", "https://tinyurl.com/abc123"]
            },
            {
                "post_id": "sample_3",
                "account": {
                    "account_age_days": 180,
                    "verified": False,
                    "historical_post_count": 50
                },
                "urls": []
            }
        ]
        
        print("Running with sample data...")
        results = analyzer.process_batch(sample_posts, "source_outputs.json")
        print("\nSample results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
