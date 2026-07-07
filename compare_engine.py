"""
Comparison Engine for Contradiction & Consensus Explorer
Analyzes multiple sources to find agreements, contradictions, and unique points
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class SourceStance(BaseModel):
    """How a source relates to a specific claim"""
    source_id: str
    stance: Literal["supports", "contradicts", "mentions", "silent"]


class CompareClaim(BaseModel):
    """A claim with per-source stances"""
    id: str
    statement: str
    stances: List[SourceStance]


class CompareResult(BaseModel):
    """Complete comparison analysis result"""
    claims: List[CompareClaim]
    summary: str
    notes: Optional[str] = None


class CompareRequest(BaseModel):
    """Request for multi-source comparison"""
    source_ids: List[str]
    query: Optional[str] = None


class ComparisonEngine:
    """Engine for analyzing multiple sources and finding contradictions/consensus"""
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.uploads_dir = workspace_dir / "uploads"
        self.transcripts_dir = workspace_dir / "transcripts"
        self.sources_file = workspace_dir / "sources.json"
    
    def load_sources_for_comparison(self, source_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Load content and metadata for selected sources"""
        # Load all sources
        try:
            all_sources = json.loads(self.sources_file.read_text(encoding="utf-8"))
        except Exception:
            all_sources = []
        
        # Create source lookup
        source_lookup = {str(s.get("id", "")): s for s in all_sources}
        
        loaded_sources = {}
        for source_id in source_ids:
            if source_id not in source_lookup:
                continue
                
            source_info = source_lookup[source_id]
            content = ""
            
            # Load content based on source type
            if source_info.get("type") == "document" and source_info.get("path"):
                try:
                    file_path = self.workspace_dir / source_info["path"]
                    if file_path.exists():
                        if file_path.suffix.lower() == ".docx":
                            import docx
                            doc = docx.Document(file_path)
                            content = "\n".join(par.text for par in doc.paragraphs)
                        else:
                            content = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    content = ""
            
            elif source_info.get("type") == "youtube" and source_info.get("id"):
                try:
                    transcript_path = self.transcripts_dir / f"{source_info['id']}.txt"
                    if transcript_path.exists():
                        content = transcript_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    content = ""
            
            # Truncate very long content to avoid context window issues
            if len(content) > 8000:
                content = content[:8000] + "\n[Content truncated for analysis]"
            
            loaded_sources[source_id] = {
                "info": source_info,
                "content": content.strip(),
                "name": source_info.get("name", f"Source {source_id}")
            }
        
        return loaded_sources
    
    def build_comparison_prompt(self, sources: Dict[str, Dict[str, Any]], query: Optional[str] = None) -> str:
        """Build a structured prompt for the LLM to analyze sources"""
        
        # Build source content section
        source_sections = []
        for source_id, source_data in sources.items():
            content = source_data["content"]
            name = source_data["name"]
            
            if not content:
                source_sections.append(f"Source {source_id} ({name}): [No content available]")
            else:
                # Limit content length per source
                max_content = 3000
                if len(content) > max_content:
                    content = content[:max_content] + "..."
                
                source_sections.append(f"Source {source_id} ({name}):\n{content}")
        
        source_text = "\n\n---\n\n".join(source_sections)
        
        # Build the main prompt
        prompt = """You are an expert analyst comparing multiple documents to identify agreements, contradictions, and unique perspectives.

Your task is to:
1. Extract key claims, statements, or positions from each source
2. Identify where sources agree with each other
3. Identify where sources contradict each other  
4. Note unique points mentioned by only some sources
5. Provide a clear summary of the overall comparison

ANALYSIS RULES:
- Focus on substantive claims, facts, and positions (not minor wording differences)
- A "contradiction" means sources make opposing claims about the same topic
- "Supports" means sources make compatible or reinforcing claims
- "Mentions" means source discusses the topic but doesn't take a clear stance
- "Silent" means source doesn't address the topic at all
- Be precise about what each source actually says

SOURCES TO ANALYZE:
""" + source_text + """

"""
        
        if query:
            prompt += f"""FOCUS TOPIC: {query}

Analyze the sources specifically with respect to this topic, but also identify other important agreements/contradictions.

"""
        
        prompt += """Return your analysis in this exact JSON format:
{
  "claims": [
    {
      "id": "claim_1",
      "statement": "Clear, specific claim statement (10-25 words)",
      "stances": [
        {"source_id": "ID", "stance": "supports|contradicts|mentions|silent"}
      ]
    }
  ],
  "summary": "2-3 sentence summary of key agreements and contradictions found",
  "notes": "Optional notes about limitations or observations"
}

Important: Return ONLY the JSON, no additional text."""
        
        return prompt
    
    def parse_comparison_response(self, response_text: str) -> CompareResult:
        """Parse LLM response into CompareResult, with fallback for malformed responses"""
        
        def safe_json_loads(text: str) -> Any:
            t = (text or "").strip()
            if not t:
                return None
            
            # Try to extract JSON object/array if model added prose
            first_brace = min([i for i in [t.find("{"), t.find("[")] if i != -1], default=-1)
            if first_brace > 0:
                t = t[first_brace:]
            last_brace = max(t.rfind("}"), t.rfind("]"))
            if last_brace != -1:
                t = t[: last_brace + 1]
            
            try:
                return json.loads(t)
            except Exception:
                return None
        
        parsed = safe_json_loads(response_text)
        
        if not parsed or not isinstance(parsed, dict):
            # Fallback for malformed responses
            return CompareResult(
                claims=[],
                summary="Unable to parse comparison analysis. The sources may need to be clearer or more distinct.",
                notes="AI response could not be parsed into structured format."
            )
        
        try:
            # Extract and validate claims
            claims_data = parsed.get("claims", [])
            claims = []
            
            for i, claim_data in enumerate(claims_data):
                if not isinstance(claim_data, dict):
                    continue
                
                claim_id = claim_data.get("id", f"claim_{i+1}")
                statement = str(claim_data.get("statement", "")).strip()
                
                if not statement:
                    continue
                
                # Extract stances
                stances_data = claim_data.get("stances", [])
                stances = []
                
                for stance_data in stances_data:
                    if not isinstance(stance_data, dict):
                        continue
                    
                    source_id = str(stance_data.get("source_id", ""))
                    stance = stance_data.get("stance", "")
                    
                    if not source_id or stance not in ["supports", "contradicts", "mentions", "silent"]:
                        continue
                    
                    stances.append(SourceStance(source_id=source_id, stance=stance))
                
                if stances:  # Only include claims with at least one stance
                    claims.append(CompareClaim(id=claim_id, statement=statement, stances=stances))
            
            summary = str(parsed.get("summary", "Comparison analysis completed.")).strip()
            notes = parsed.get("notes")
            if notes:
                notes = str(notes).strip()
            
            return CompareResult(claims=claims, summary=summary, notes=notes)
            
        except Exception as e:
            # Fallback for parsing errors
            return CompareResult(
                claims=[],
                summary="Error occurred while parsing comparison analysis.",
                notes=f"Parsing error: {str(e)}"
            )
    
    def analyze_sources(self, source_ids: List[str], query: Optional[str] = None, llm_generate_func=None) -> CompareResult:
        """Main function to analyze multiple sources for contradictions and consensus"""
        
        # Validate inputs
        if not source_ids:
            return CompareResult(
                claims=[],
                summary="No sources provided for comparison.",
                notes="Please select at least 2 sources to compare."
            )
        
        if len(source_ids) < 2:
            return CompareResult(
                claims=[],
                summary="Need at least 2 sources for meaningful comparison.",
                notes="Comparison requires multiple sources to find agreements and contradictions."
            )
        
        # Load sources
        sources = self.load_sources_for_comparison(source_ids)
        
        if len(sources) < 2:
            return CompareResult(
                claims=[],
                summary="Not enough valid sources found for comparison.",
                notes=f"Only {len(sources)} sources could be loaded. Check source availability."
            )
        
        # Check if we have meaningful content
        content_sources = [s for s in sources.values() if s["content"]]
        if len(content_sources) < 2:
            return CompareResult(
                claims=[],
                summary="Not enough content available for comparison.",
                notes="At least 2 sources need readable content for analysis."
            )
        
        # Build prompt
        prompt = self.build_comparison_prompt(sources, query)
        
        # Get LLM analysis
        if llm_generate_func is None:
            return CompareResult(
                claims=[],
                summary="LLM analysis function not provided.",
                notes="Internal error: missing analysis function."
            )
        
        try:
            response = llm_generate_func(prompt)
            return self.parse_comparison_response(response)
        except Exception as e:
            return CompareResult(
                claims=[],
                summary="Analysis failed due to an error.",
                notes=f"Analysis error: {str(e)}"
            )
