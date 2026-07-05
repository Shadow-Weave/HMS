"""
LLM-driven query rewriting strategy.

This module implements a query rewriting strategy that uses LLM to:
1. Intelligently determine if query expansion is needed
2. Determine if strict temporal filtering is needed
3. Generate entity expansions (subcategories, synonyms, related concepts)
4. Calculate precise time windows based on relative time expressions
5. Rewrite the original query with expanded content
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .strategies import QueryRewritingStrategy, query_rewriting_registry

logger = logging.getLogger(__name__)


class QueryAnalysisResult:
    """
    Result of LLM-driven query analysis.
    
    Attributes:
        needs_expansion: Whether the query needs entity expansion
        needs_time_window: Whether the query needs strict temporal filtering
        rewritten_query: The rewritten query with expanded content
        time_window_start: Start date of the time window (if applicable)
        time_window_end: End date of the time window (if applicable)
        expanded_entities: List of expanded entity terms
        confidence: Confidence score (0-1) for the analysis
    """
    
    def __init__(
        self,
        needs_expansion: bool = False,
        needs_time_window: bool = False,
        rewritten_query: str = "",
        time_window_start: Optional[datetime] = None,
        time_window_end: Optional[datetime] = None,
        expanded_entities: Optional[List[str]] = None,
        confidence: float = 0.0,
    ):
        self.needs_expansion = needs_expansion
        self.needs_time_window = needs_time_window
        self.rewritten_query = rewritten_query
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.expanded_entities = expanded_entities or []
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "needs_expansion": self.needs_expansion,
            "needs_time_window": self.needs_time_window,
            "rewritten_query": self.rewritten_query,
            "time_window_start": self.time_window_start.isoformat() if self.time_window_start else None,
            "time_window_end": self.time_window_end.isoformat() if self.time_window_end else None,
            "expanded_entities": self.expanded_entities,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryAnalysisResult":
        """Create from dictionary."""
        return cls(
            needs_expansion=data.get("needs_expansion", False),
            needs_time_window=data.get("needs_time_window", False),
            rewritten_query=data.get("rewritten_query", ""),
            time_window_start=datetime.fromisoformat(data["time_window_start"]) if data.get("time_window_start") else None,
            time_window_end=datetime.fromisoformat(data["time_window_end"]) if data.get("time_window_end") else None,
            expanded_entities=data.get("expanded_entities", []),
            confidence=data.get("confidence", 0.0),
        )


def _build_query_analysis_prompt(query_text: str, question_date: Optional[datetime] = None) -> List[Dict[str, str]]:
    """
    Build the prompt for LLM-based query analysis.
    
    Args:
        query_text: The original user query
        question_date: Optional date when the question was asked
        
    Returns:
        List of messages for LLM call
    """
    current_time = question_date or datetime.now(timezone.utc)
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    system_prompt = f"""
You are an intelligent query analyzer for a memory retrieval system.
Your task is to analyze user questions and provide structured analysis for improved retrieval.

CURRENT TIME: {current_time_str}

OUTPUT FORMAT:
Return ONLY a valid JSON object with these fields:
- "needs_expansion": boolean - True if the query would benefit from entity/concept expansion
- "needs_time_window": boolean - True if the query contains temporal references requiring precise filtering
- "rewritten_query": string - The rewritten query with expanded entities and resolved time references
- "time_window_start": string or null - ISO 8601 format start date (e.g., "2025-06-01T00:00:00+00:00") or null
- "time_window_end": string or null - ISO 8601 format end date (e.g., "2025-06-30T23:59:59+00:00") or null
- "expanded_entities": array of strings - List of expanded entity terms (subcategories, synonyms, related concepts)
- "confidence": number - Confidence score (0.0-1.0)

ANALYSIS GUIDELINES:

1. needs_expansion:
   - True for questions about categories (e.g., "appliances", "activities")
   - True for "how many" questions about items/things
   - True for questions with abstract concepts that could have multiple expressions
   - False for very specific questions (e.g., "What is my mother's birthday?")

2. needs_time_window:
   - True if question contains time references (yesterday, last week, January, etc.)
   - True for questions about events within a specific period
   - False for timeless questions or questions about permanent facts

3. rewritten_query:
   - Resolve relative time expressions to absolute dates
   - Expand entity categories to include subcategories
   - Include synonyms and related concepts
   - Keep the original intent intact
   - Example: "I bought several appliances last month" -> "I bought several appliances (including kitchen appliances, home appliances, electronic devices) in July 2025"

4. time_window_start/end:
   - Convert relative time to absolute date range
   - Use UTC timezone
   - "last month" -> start: first day of previous month at 00:00, end: last day of previous month at 23:59
   - "yesterday" -> start: yesterday at 00:00, end: yesterday at 23:59
   - "last week" -> start: Monday of last week at 00:00, end: Sunday of last week at 23:59

5. expanded_entities:
   - Include subcategories (e.g., "appliances" -> ["kitchen appliances", "washing machine", "refrigerator"])
   - Include synonyms (e.g., "buy" -> ["purchase", "acquire", "get"])
   - Include related concepts that might appear in stored memories
   - Maximum 8 expanded terms

6. confidence:
   - 1.0 for clear, unambiguous questions
   - Lower for vague or ambiguous questions

EXAMPLES:

Input: "How many electronic devices did I buy last month?"
Output: {{
  "needs_expansion": true,
  "needs_time_window": true,
  "rewritten_query": "How many electronic devices (including smartphones, laptops, tablets, headphones, smartwatches) did I buy in July 2025?",
  "time_window_start": "2025-07-01T00:00:00+00:00",
  "time_window_end": "2025-07-31T23:59:59+00:00",
  "expanded_entities": ["smartphones", "laptops", "tablets", "headphones", "smartwatches", "purchase", "acquire"],
  "confidence": 0.95
}}

Input: "What is my favorite color?"
Output: {{
  "needs_expansion": false,
  "needs_time_window": false,
  "rewritten_query": "What is my favorite color?",
  "time_window_start": null,
  "time_window_end": null,
  "expanded_entities": [],
  "confidence": 1.0
}}

Input: "I attended several conferences last year. How many were about AI?"
Output: {{
  "needs_expansion": true,
  "needs_time_window": true,
  "rewritten_query": "I attended several conferences (including seminars, workshops, symposiums) in 2024. How many were about AI, artificial intelligence, machine learning?",
  "time_window_start": "2024-01-01T00:00:00+00:00",
  "time_window_end": "2024-12-31T23:59:59+00:00",
  "expanded_entities": ["seminars", "workshops", "symposiums", "AI", "artificial intelligence", "machine learning"],
  "confidence": 0.9
}}

IMPORTANT: Return ONLY the JSON object, NO additional text or explanations!
""".strip()
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this query:\n{query_text}"},
    ]


def _parse_analysis_response(response_text: str) -> QueryAnalysisResult:
    """
    Parse the LLM response into QueryAnalysisResult.
    
    Args:
        response_text: Raw LLM response text
        
    Returns:
        Parsed QueryAnalysisResult
    """
    try:
        # Clean response - sometimes LLM includes extra text
        # Find the first { and last }
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            logger.warning(f"Failed to find JSON in response: {response_text[:100]}...")
            return QueryAnalysisResult(confidence=0.0)
        
        json_str = response_text[start_idx:end_idx]
        data = json.loads(json_str)
        
        # Parse datetime strings
        time_window_start = None
        if data.get("time_window_start"):
            try:
                time_window_start = datetime.fromisoformat(data["time_window_start"].replace("Z", "+00:00"))
            except ValueError:
                logger.warning(f"Invalid time_window_start format: {data['time_window_start']}")
        
        time_window_end = None
        if data.get("time_window_end"):
            try:
                time_window_end = datetime.fromisoformat(data["time_window_end"].replace("Z", "+00:00"))
            except ValueError:
                logger.warning(f"Invalid time_window_end format: {data['time_window_end']}")
        
        return QueryAnalysisResult(
            needs_expansion=data.get("needs_expansion", False),
            needs_time_window=data.get("needs_time_window", False),
            rewritten_query=data.get("rewritten_query", ""),
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            expanded_entities=data.get("expanded_entities", []),
            confidence=data.get("confidence", 0.0),
        )
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse analysis response: {e}")
        return QueryAnalysisResult(confidence=0.0)
    except Exception as e:
        logger.error(f"Unexpected error parsing analysis response: {e}")
        return QueryAnalysisResult(confidence=0.0)


@query_rewriting_registry.register("llm_driven")
class LLMDrivenQueryRewriting(QueryRewritingStrategy):
    """
    LLM-driven query rewriting strategy.
    
    Uses an LLM to:
    1. Intelligently determine if query expansion is needed
    2. Determine if strict temporal filtering is needed
    3. Generate entity expansions and calculate time windows
    4. Rewrite the query with expanded content
    
    This replaces the rule-based approach with AI-driven analysis.
    """
    
    def __init__(
        self,
        min_query_length: int = 5,
        max_retries: int = 2,
        cache_enabled: bool = True,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize LLM-driven query rewriting.
        
        Args:
            min_query_length: Minimum query length to consider for expansion
            max_retries: Maximum retry attempts on LLM failure
            cache_enabled: Whether to cache results per query
            confidence_threshold: Minimum confidence score to use results
        """
        self._min_query_length = min_query_length
        self._max_retries = max_retries
        self._cache: Dict[str, QueryAnalysisResult] = {}
        self._cache_enabled = cache_enabled
        self._confidence_threshold = confidence_threshold
    
    @property
    def name(self) -> str:
        return "llm_driven"
    
    async def analyze(
        self,
        query_text: str,
        llm: Optional[Any] = None,
        question_date: Optional[datetime] = None,
    ) -> QueryAnalysisResult:
        """
        Analyze query using LLM and return structured analysis result.
        
        Args:
            query_text: Original user query
            llm: LLM provider instance
            question_date: Optional date when the question was asked
            
        Returns:
            QueryAnalysisResult with all analysis results
        """
        # Check minimum length
        if len(query_text) < self._min_query_length:
            return QueryAnalysisResult(
                needs_expansion=False,
                needs_time_window=False,
                rewritten_query=query_text,
                confidence=1.0,
            )
        
        # Check cache
        cache_key = f"{query_text}_{question_date}" if question_date else query_text
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Check LLM availability
        if llm is None:
            logger.warning("LLM not available for query analysis, returning default")
            return QueryAnalysisResult(
                needs_expansion=False,
                needs_time_window=False,
                rewritten_query=query_text,
                confidence=0.5,
            )
        
        # Build prompt and call LLM
        messages = _build_query_analysis_prompt(query_text, question_date)
        
        for attempt in range(self._max_retries):
            try:
                response = await llm.call(
                    messages=messages,
                    temperature=0.1,
                    max_completion_tokens=1024,
                )
                
                result = _parse_analysis_response(response)
                
                # Validate and fallback if confidence is too low
                if result.confidence < self._confidence_threshold:
                    logger.warning(f"Analysis confidence {result.confidence} below threshold {self._confidence_threshold}")
                    result = QueryAnalysisResult(
                        needs_expansion=False,
                        needs_time_window=False,
                        rewritten_query=query_text,
                        confidence=0.5,
                    )
                
                # Cache the result
                if self._cache_enabled:
                    self._cache[cache_key] = result
                
                return result
                
            except Exception as e:
                logger.warning(f"LLM analysis attempt {attempt + 1} failed: {e}")
                if attempt == self._max_retries - 1:
                    # Return fallback result on final failure
                    return QueryAnalysisResult(
                        needs_expansion=False,
                        needs_time_window=False,
                        rewritten_query=query_text,
                        confidence=0.0,
                    )
        
        # Shouldn't reach here, but just in case
        return QueryAnalysisResult(
            needs_expansion=False,
            needs_time_window=False,
            rewritten_query=query_text,
            confidence=0.0,
        )
    
    async def rewrite(
        self,
        query_text: str,
        llm: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, List[str]]:
        """
        Rewrite query using LLM for alias expansion with confidence filtering.
        
        This is the standard interface method that returns the legacy format.
        
        Args:
            query_text: Original user query
            llm: LLM provider instance
            **kwargs: Additional context (may contain 'question_date')
            
        Returns:
            Dict mapping original query -> expanded aliases
        """
        question_date = kwargs.get("question_date")
        result = await self.analyze(query_text, llm, question_date)
        
        if result.needs_expansion and result.expanded_entities:
            return {query_text: result.expanded_entities}
        
        return {query_text: []}
    
    def should_expand(self, query_text: str) -> bool:
        """
        Legacy method - always returns True since we let LLM decide.
        
        The actual decision is made in the analyze() method.
        """
        return len(query_text) >= self._min_query_length
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)
