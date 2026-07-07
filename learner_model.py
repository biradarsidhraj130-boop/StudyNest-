"""
Learner Model Module for Cognitive Twin Feature
Tracks user performance at concept level across quizzes and flashcards
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QuizResult(BaseModel):
    """Single quiz question result"""
    quiz_id: str
    question: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    concepts: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FlashcardResult(BaseModel):
    """Single flashcard study result"""
    card_id: str
    front: str
    back: str
    rating: int  # 1-5 scale (1=hard, 5=easy)
    concepts: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConceptMastery(BaseModel):
    """Mastery tracking for a single concept"""
    mastery_level: float = Field(default=0.0, ge=0.0, le=1.0)
    correct_count: int = Field(default=0, ge=0)
    total_count: int = Field(default=0, ge=0)
    last_seen: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sources: List[str] = Field(default_factory=list)  # ["quiz", "flashcards"]


class Recommendation(BaseModel):
    """Study recommendation for a concept"""
    concept: str
    reason: str  # "low_mastery", "not_seen_recently", "struggling"
    suggested_action: str  # "quiz", "flashcards", "review"
    priority: int = Field(default=1, ge=1, le=5)  # 1=highest priority


class LearnerModel(BaseModel):
    """Complete learner model data structure"""
    user_id: str = "default"
    concepts: Dict[str, ConceptMastery] = Field(default_factory=dict)
    quiz_history: List[QuizResult] = Field(default_factory=list)
    flashcard_history: List[FlashcardResult] = Field(default_factory=list)
    recommendations: Dict[str, List[Recommendation]] = Field(default_factory=lambda: {"daily": []})
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LearnerModelManager:
    """Manages learner model persistence and operations"""
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.learner_model_file = workspace_dir / "learner_model.json"
    
    def load_learner_model(self) -> LearnerModel:
        """Load learner model from file, create default if doesn't exist"""
        if not self.learner_model_file.exists():
            return LearnerModel()
        
        try:
            data = json.loads(self.learner_model_file.read_text(encoding="utf-8"))
            return LearnerModel(**data)
        except Exception:
            # If file is corrupted, start fresh
            return LearnerModel()
    
    def save_learner_model(self, model: LearnerModel) -> None:
        """Save learner model to file"""
        model.last_updated = datetime.now(timezone.utc).isoformat()
        self.learner_model_file.write_text(
            json.dumps(model.dict(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def extract_concepts_from_content(self, content: str, content_type: str = "quiz") -> List[str]:
        """
        Extract key concepts from quiz questions or flashcard content.
        For now, use simple keyword extraction. Can be enhanced with NVIDIA NIM later.
        """
        # Simple concept extraction - look for important terms
        # This is a basic implementation that can be enhanced
        import re
        
        # Common technical/concept indicators
        concept_patterns = [
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b',  # CamelCase terms
            r'\b([a-z]+_[a-z_]+)\b',  # snake_case terms
            r'\b(\w+(?:tion|sion|ment|ism|logy|graphy|ics))\b',  # Academic suffixes
        ]
        
        concepts = set()
        for pattern in concept_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            concepts.update([match.lower() for match in matches if len(match) > 3])
        
        # Filter out common words
        stop_words = {
            'this', 'that', 'with', 'from', 'they', 'have', 'been', 'said', 'each',
            'which', 'their', 'time', 'will', 'about', 'if', 'would', 'there', 'could'
        }
        
        filtered_concepts = [
            concept for concept in concepts 
            if concept not in stop_words and len(concept) > 2
        ]
        
        return filtered_concepts[:5]  # Limit to top 5 concepts per item
    
    def update_mastery_levels(self, model: LearnerModel) -> None:
        """Update mastery levels based on recent performance"""
        for concept_name, mastery in model.concepts.items():
            if mastery.total_count == 0:
                continue
            
            # Calculate mastery based on recent performance
            recent_correct = 0
            recent_total = 0
            
            # Check recent quiz history (last 20 items)
            for quiz_result in model.quiz_history[-20:]:
                if concept_name in quiz_result.concepts:
                    recent_total += 1
                    if quiz_result.is_correct:
                        recent_correct += 1
            
            # Check recent flashcard history (last 20 items)
            for flash_result in model.flashcard_history[-20:]:
                if concept_name in flash_result.concepts:
                    recent_total += 1
                    if flash_result.rating >= 3:  # Rating 3+ is considered correct
                        recent_correct += 1
            
            # Update mastery level with exponential moving average
            if recent_total > 0:
                recent_accuracy = recent_correct / recent_total
                # Weight recent performance more heavily
                new_mastery = (mastery.mastery_level * 0.3) + (recent_accuracy * 0.7)
                mastery.mastery_level = max(0.0, min(1.0, new_mastery))
    
    def generate_recommendations(self, model: LearnerModel) -> List[Recommendation]:
        """Generate study recommendations based on mastery levels"""
        recommendations = []
        
        for concept_name, mastery in model.concepts.items():
            # Recommend concepts with low mastery
            if mastery.mastery_level < 0.6:
                priority = 1 if mastery.mastery_level < 0.3 else 2
                recommendations.append(Recommendation(
                    concept=concept_name,
                    reason="low_mastery",
                    suggested_action="quiz",
                    priority=priority
                ))
            
            # Recommend concepts not seen recently
            elif mastery.last_seen:
                days_since_seen = (datetime.now(timezone.utc) - 
                                 datetime.fromisoformat(mastery.last_seen.replace('Z', '+00:00'))).days
                if days_since_seen > 7:
                    recommendations.append(Recommendation(
                        concept=concept_name,
                        reason="not_seen_recently",
                        suggested_action="flashcards",
                        priority=3
                    ))
        
        # Sort by priority and limit to top 5
        recommendations.sort(key=lambda r: r.priority)
        return recommendations[:5]
    
    def log_quiz_results(self, model: LearnerModel, quiz_data: List[Dict[str, Any]], 
                        user_answers: Dict[int, str]) -> None:
        """Log quiz completion results"""
        quiz_id = f"quiz_{int(time.time() * 1000)}"
        
        for idx, question_data in enumerate(quiz_data):
            if idx >= len(user_answers):
                continue
                
            question = question_data.get("question", "")
            correct_answer = question_data.get("answer", "")
            user_answer = user_answers.get(idx, "")
            is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
            
            # Extract concepts
            concepts = self.extract_concepts_from_content(question, "quiz")
            
            # Create quiz result
            quiz_result = QuizResult(
                quiz_id=quiz_id,
                question=question,
                user_answer=user_answer,
                correct_answer=correct_answer,
                is_correct=is_correct,
                concepts=concepts
            )
            
            model.quiz_history.append(quiz_result)
            
            # Update concept mastery
            for concept in concepts:
                if concept not in model.concepts:
                    model.concepts[concept] = ConceptMastery()
                
                mastery = model.concepts[concept]
                mastery.total_count += 1
                if is_correct:
                    mastery.correct_count += 1
                if "quiz" not in mastery.sources:
                    mastery.sources.append("quiz")
                mastery.last_seen = quiz_result.timestamp
        
        # Update mastery levels and recommendations
        self.update_mastery_levels(model)
        model.recommendations["daily"] = self.generate_recommendations(model)
    
    def log_flashcard_results(self, model: LearnerModel, flashcard_data: List[Dict[str, Any]], 
                             ratings: Dict[int, int]) -> None:
        """Log flashcard study results"""
        study_session_id = f"flashcards_{int(time.time() * 1000)}"
        
        for idx, card_data in enumerate(flashcard_data):
            if idx >= len(ratings):
                continue
                
            front = card_data.get("front", "")
            back = card_data.get("back", "")
            rating = ratings.get(idx, 3)  # Default to neutral rating
            
            # Extract concepts from both front and back
            concepts = self.extract_concepts_from_content(f"{front} {back}", "flashcards")
            
            # Create flashcard result
            flashcard_result = FlashcardResult(
                card_id=f"{study_session_id}_{idx}",
                front=front,
                back=back,
                rating=rating,
                concepts=concepts
            )
            
            model.flashcard_history.append(flashcard_result)
            
            # Update concept mastery
            for concept in concepts:
                if concept not in model.concepts:
                    model.concepts[concept] = ConceptMastery()
                
                mastery = model.concepts[concept]
                mastery.total_count += 1
                if rating >= 3:  # Rating 3+ is considered correct
                    mastery.correct_count += 1
                if "flashcards" not in mastery.sources:
                    mastery.sources.append("flashcards")
                mastery.last_seen = flashcard_result.timestamp
        
        # Update mastery levels and recommendations
        self.update_mastery_levels(model)
        model.recommendations["daily"] = self.generate_recommendations(model)
