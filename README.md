# Knowledge Workspace - Complete Documentation

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features A-Z](#features-a-z)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Backend API Reference](#backend-api-reference)
- [Frontend Architecture](#frontend-architecture)
- [Document Ingestion System](#document-ingestion-system)
- [Configuration](#configuration)
- [Development Guide](#development-guide)
- [Troubleshooting](#troubleshooting)

---

## Overview

Knowledge Workspace is a comprehensive AI-powered learning and document management platform. It combines **Retrieval-Augmented Generation (RAG)** with an intelligent **Cognitive Twin** system that tracks your learning progress, identifies weak concepts, and provides personalized recommendations.

### Key Capabilities
- 📚 **Universal Document Support** - PDF, DOCX, PPTX, XLSX, TXT, HTML, YouTube videos, images with OCR
- 🤖 **AI-Powered Chat** - Ask questions based on your documents using NVIDIA NIM LLM
- 🎯 **Cognitive Twin** - Tracks learning patterns, quiz scores, identifies weak concepts
- 🔍 **Smart Discovery** - Find and import external learning resources (YouTube, PDFs, web articles)
- 📊 **Learning Tools** - Auto-generated summaries, quizzes, flashcards, mind maps, infographics
- 📋 **Curriculum Builder** - Auto-build personalized learning paths from your sources
- ⚖️ **Source Comparison** - Compare multiple sources for contradictions and agreements

---

## Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + Vite)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │ ChatPage │ │DiscoverPg│ │ QuizPage │ │Flashcards│         │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │ MindMap  │ │Infographc│ │Curriculum│ │Learning  │         │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST + WebSocket
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI + Python)                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Core API Layer (app.py)                       │  │
│  │  /upload    /ask    /summary    /quiz    /flashcards      │  │
│  │  /mindmap   /infographic  /youtube   /reset             │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │ Document    │ │ Learner     │ │ Curriculum  │              │
│  │ Ingestion   │ │ Model       │ │ Builder     │              │
│  │ System      │ │ Manager     │ │ Engine      │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │ Comparison  │ │ Recommendation│ │ Import      │              │
│  │ Engine      │ │ Engine      │ │ Handler     │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                          │
│  NVIDIA NIM API (LLM)    │    YouTube Data API               │
│  - LLaMA 3.1 70B         │    - Transcript extraction        │
│  - Text generation       │    - Video metadata               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **Document Upload** → Extract text using OCR/engine-specific extractors → Store in sources.json
2. **User Query** → Retrieve relevant sources → Build context prompt → NVIDIA LLM → Response
3. **Learning Activity** → Log to learner model → Update concept scores → Generate recommendations
4. **Discover Search** → Query recommendation engine → Return filtered results → Import to workspace

---

## Features A-Z

### A - Ask (Chat)
Natural language Q&A based on uploaded documents. Uses RAG to provide grounded answers with citations.

### B - Backend APIs
Complete REST API suite for all operations. See [Backend API Reference](#backend-api-reference).

### C - Cognitive Twin
Personalized learning profile that tracks:
- Quiz scores per concept
- Flashcard confidence ratings
- Study session duration
- Knowledge mastery levels (0-1 scale)

### D - Discover
Find external learning resources:
- **Search by Topic** - Query for any subject, filter by type/difficulty/duration
- **Find Similar** - Analyze uploaded source → extract topics → find related content
- **One-Click Import** - Add YouTube videos, PDFs, web articles to workspace

### E - Export/Extract
Universal document extraction supporting PDF, DOCX, PPTX, XLSX, TXT, HTML, images (OCR).

### F - Flashcards
Auto-generated study cards with spaced repetition. Tracks confidence ratings (1-5) to identify weak areas.

### G - Generation Tools
- **Summaries** - Concise or detailed document summaries
- **Quizzes** - Multiple-choice questions with explanations
- **Flashcards** - Front/back study cards
- **Mind Maps** - Visual concept relationship graphs
- **Infographics** - Structured visual summaries

### H - History & Tracking
Complete learning activity history with timestamps and performance metrics.

### I - Import System
Import recommended resources:
- YouTube videos (auto-transcript)
- PDFs (auto-download and extract)
- Web articles (content scraping)

### J - JSON Storage
All data stored in workspace_data/:
- sources.json - Document metadata
- learner_model.json - Learning profile
- curriculum.json - Generated learning paths
- transcripts/ - YouTube text
- uploads/ - Document files

### K - Knowledge Graph
Mind map generation showing concept relationships and dependencies.

### L - Learning Profile
Dashboard showing:
- Concept mastery levels
- Weak areas requiring attention
- Study streaks and activity
- Personalized recommendations

### M - Mind Maps
Visual concept mapping with nodes and edges, automatically generated from document content.

### N - NVIDIA NIM Integration
LLM powered by NVIDIA NIM API:
- Model: abacusai/dracarys-llama-3.1-70b-instruct
- Temperature: 0.2
- Max tokens: 2000
- 60-second timeout

### O - OCR (Optical Character Recognition)
Fallback extraction for scanned PDFs and images using Tesseract OCR.

### P - Progress Tracking
Real-time progress indicators for:
- File uploads
- YouTube transcript processing
- Document text extraction

### Q - Quiz Generation
Auto-generated multiple-choice quizzes with:
- 5-10 questions
- 4 options each
- Correct answers with explanations

### R - Recommendations
Personalized resource recommendations based on:
- Weak concept areas
- Previously imported content
- Content type preferences

### S - Source Management
Add, delete, view sources with metadata extraction and status tracking.

### T - Transcription
Automatic YouTube transcript extraction using yt-dlp and youtube-transcript-api.

### U - Universal Document Support
See [Document Ingestion System](#document-ingestion-system).

### V - Visualization
Rich visual outputs:
- Mind maps (node-edge graphs)
- Infographics (structured items)
- Progress charts

### W - Workspace Management
Multi-user workspace with persistent storage and session management.

### X - eXtensible Architecture
Plugin-based design for:
- New document extractors
- Additional LLM providers
- Custom recommendation sources

### Y - YouTube Integration
Add YouTube videos by URL:
- Auto-extract transcript
- Video metadata
- Processing status tracking

### Z - Zustand State Management
Frontend uses Zustand for:
- Global state management
- Source list
- Chat messages
- Loading states

---

## Tech Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2.0 | UI framework |
| TypeScript | 5.2.2 | Type safety |
| Vite | 4.5.0 | Build tool |
| TailwindCSS | 3.3.5 | Styling |
| Framer Motion | 10.16.5 | Animations |
| Zustand | 4.4.6 | State management |
| React Query | 5.8.4 | Data fetching |
| React Router | 6.30.1 | Routing |
| Lucide React | 0.294.0 | Icons |
| React Markdown | 9.0.1 | Markdown rendering |
| Axios | 1.6.0 | HTTP client |

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.115.0 | Web framework |
| Uvicorn | 0.30.6 | ASGI server |
| Pydantic | 2.8.2 | Data validation |
| Requests | 2.32.3 | HTTP client |
| PyMuPDF | 1.24.5 | PDF processing |
| python-docx | 1.2.0 | Word documents |
| python-pptx | 0.6.23 | PowerPoint |
| openpyxl | 3.1.5 | Excel files |
| youtube-transcript-api | 0.6.2 | YouTube transcripts |
| yt-dlp | 2025.1.26 | YouTube downloads |
| pytesseract | 0.3.13 | OCR |
| Pillow | 10.3.0 | Image processing |
| pdf2image | 1.17.0 | PDF to image |
| faster-whisper | 1.1.0 | Audio transcription |

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- NVIDIA API Key (for LLM features)

### 1. Clone and Setup Backend

```bash
cd knowledge_workspace/backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "NVIDIA_API_KEY=your_api_key_here" > .env
```

### 2. Setup Frontend

```bash
cd knowledge_workspace/frontend

# Install dependencies
npm install

# Create .env file
echo "VITE_API_BASE_URL=http://127.0.0.1:8000" > .env
```

### 3. Run the Application

```bash
# Terminal 1 - Backend
cd knowledge_workspace/backend
.\venv\Scripts\activate
python app.py

# Terminal 2 - Frontend
cd knowledge_workspace/frontend
npm run dev
```

### 4. Access the Application

- Frontend: http://localhost:5173
- Backend API: http://127.0.0.1:8000
- API Docs: http://127.0.0.1:8000/docs

---

## Backend API Reference

### Core Endpoints

#### Document Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload document file |
| POST | `/youtube` | Add YouTube video by URL |
| GET | `/sources` | List all sources |
| DELETE | `/sources` | Delete source by filename |
| POST | `/reset` | Clear all sources |

#### Chat & Generation
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ask` | Chat with LLM using document context |
| POST | `/summary` | Generate document summary |
| POST | `/quiz` | Generate quiz questions |
| POST | `/flashcards` | Generate flashcards |
| POST | `/mindmap` | Generate mind map |
| POST | `/infographic` | Generate infographic |

#### Learning Model
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/learner-model` | Get learning profile |
| POST | `/learner-model/log-quiz` | Log quiz results |
| POST | `/learner-model/log-flashcard` | Log flashcard ratings |
| GET | `/learner-model/recommendations` | Get study recommendations |

#### Curriculum
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/curriculum` | Get or build curriculum |
| POST | `/curriculum/build` | Force rebuild curriculum |

#### Comparison
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/compare/sources` | Compare multiple sources |

#### Discovery & Recommendations
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/recommend` | Get resource recommendations by topic |
| POST | `/recommend/from-source` | Find similar to uploaded source |
| POST | `/recommend/import` | Import recommended resource |

### Request/Response Examples

#### Upload Document
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@document.pdf"
```

Response:
```json
{
  "ok": true,
  "stats": {
    "bytes": 12345,
    "extracted": true,
    "text_length": 5000
  }
}
```

#### Ask Question
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "top_k": 4}'
```

Response:
```json
{
  "answer": "The main topic is...",
  "sources": [...]
}
```

#### Get Recommendations
```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "machine learning",
    "max_items": 10,
    "type_filter": ["youtube", "pdf"],
    "difficulty": "beginner"
  }'
```

---

## Frontend Architecture

### Project Structure
```
frontend/src/
├── api/                    # API clients
│   ├── client.ts          # Axios instance
│   ├── recommendationApi.ts
│   ├── curriculumApi.ts
│   ├── compareApi.ts
│   └── learnerModelApi.ts
├── app/
│   ├── layouts/
│   │   └── AppShell.tsx   # Main app layout
│   ├── providers/
│   │   ├── store.ts      # Zustand store
│   │   ├── theme.tsx     # Theme provider
│   │   └── react-query.tsx
│   └── router/
│       └── AppRouter.tsx  # Route configuration
├── components/
│   ├── loaders/
│   └── shared/
│       ├── Sidebar.tsx
│       ├── SourcePanel.tsx
│       └── CitationsPanel.tsx
├── hooks/
│   └── useWorkspaceApi.ts  # API hooks
├── pages/
│   ├── ChatPage.tsx       # Main chat interface
│   ├── DiscoverPage.tsx   # Resource discovery
│   ├── QuizPage.tsx       # Quiz generation
│   ├── FlashcardsPage.tsx
│   ├── MindmapPage.tsx
│   ├── InfographicPage.tsx
│   ├── CurriculumPage.tsx
│   ├── LearningProfilePage.tsx
│   ├── CompareSourcesPage.tsx
│   ├── YouTubePage.tsx
│   ├── SummaryPage.tsx
│   └── LoginPage.tsx
└── styles/
    └── globals.css
```

### State Management (Zustand)
```typescript
// Key store properties
interface AppState {
  sources: Source[]           // Uploaded documents
  messages: Message[]         // Chat history
  isLoading: boolean
  errorText: string | null
  lastCitations: Citation[] | null
  llmProgress: { percent: number; label: string } | null
  isAuthed: boolean
}
```

### Key Custom Hooks
- `useWorkspaceApi` - Document upload, YouTube add, chat, quiz, etc.
- `useQuery` (React Query) - Recommendation fetching with caching

---

## Document Ingestion System

### Supported File Types
| Type | Extension | Extraction Method |
|------|-----------|-------------------|
| PDF | .pdf | PyMuPDF + OCR fallback |
| Word | .docx | python-docx |
| PowerPoint | .pptx | python-pptx |
| Excel | .xlsx | openpyxl |
| Text | .txt | Native Python |
| HTML | .html, .htm | BeautifulSoup |
| Images | .png, .jpg | Tesseract OCR |

### Extraction Process
1. **File Upload** → Save to uploads/
2. **Type Detection** → Map extension to extractor
3. **Text Extraction** → Use appropriate library
4. **OCR Fallback** → For scanned PDFs/images (if enabled)
5. **Metadata Storage** → Save to sources.json

### OCR Configuration
```bash
# Enable/disable OCR
ENABLE_OCR=true  # or false

# Linux/macOS dependencies
sudo apt-get install tesseract-ocr poppler-utils  # Ubuntu
brew install tesseract poppler                   # macOS
```

---

## Configuration

### Environment Variables

#### Backend (.env)
```bash
# Required
NVIDIA_API_KEY=nvapi-xxxxx

# Optional
NVIDIA_MODEL=abacusai/dracarys-llama-3.1-70b-instruct
NVIDIA_TIMEOUT_S=60
HOST=127.0.0.1
PORT=8000
ENABLE_OCR=true
```

#### Frontend (.env)
```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### File Locations
```
backend/workspace_data/
├── sources.json          # Source metadata
├── learner_model.json    # Learning profile
├── curriculum.json       # Generated curriculum
├── compare_cache.json    # Comparison results
├── uploads/              # Uploaded documents
├── transcripts/          # YouTube transcripts
└── temp/                 # Temporary downloads
```

---

## Development Guide

### Adding New API Endpoints

1. **Add Pydantic model** (in relevant module):
```python
class MyRequest(BaseModel):
    param: str
```

2. **Add endpoint** in app.py:
```python
@app.post("/my-endpoint")
def my_endpoint(req: MyRequest) -> dict[str, Any]:
    return {"ok": True, "result": ...}
```

3. **Add frontend API** in src/api/:
```typescript
export async function myApiCall(param: string) {
  const res = await api.post('/my-endpoint', { param })
  return res.data
}
```

4. **Add page/component** using the API

### Adding Document Extractors

1. Add extractor function in `document_ingestion.py`:
```python
def _extract_newtype(self, path: Path) -> ExtractResult:
    # Implementation
    return ExtractResult(success=True, text="...")
```

2. Register in `_get_extractor_map()`:
```python
'.newext': self._extract_newtype,
```

### Testing

```bash
# Test document ingestion
cd backend
python test_document_ingestion.py

# Test curriculum
cd backend
python test_curriculum_simple.py

# Run frontend dev server
cd frontend
npm run dev
```

---

## Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check port 8000 is free
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Kill process if needed
taskkill /PID <PID> /F         # Windows
kill -9 <PID>                 # Linux/Mac
```

#### LLM Not Responding
- Verify `NVIDIA_API_KEY` in .env
- Check NVIDIA NIM service status
- Verify internet connection

#### Document Upload Fails
- Check file size < 10MB (frontend limit)
- Verify file type is supported
- Check OCR dependencies if scanned PDF

#### YouTube Import Fails
- Video must be publicly accessible
- Check yt-dlp is installed
- Transcript must be available (or auto-generated)

#### QueryClient Error
- Ensure ReactQueryProvider wraps App in main.tsx

### Debug Logs

Backend logs show in console:
```
INFO:     127.0.0.1:xxxxx - "POST /upload HTTP/1.1" 200 OK
```

Browser console shows frontend errors:
- F12 → Console tab

---

## Project Scripts

### Backend
```bash
# Run server
python app.py

# Test scripts
python test_curriculum.py
python test_document_ingestion.py
python test_upload_ingestion.py
```

### Frontend
```bash
# Development
npm run dev

# Production build
npm run build

# Preview production build
npm run preview

# Lint
npm run lint
```

### Convenience Scripts (Windows)
```bash
# Run backend (clean)
scripts/run-knowledge-workspace-backend-clean.cmd

# Run backend (normal)
scripts/run-knowledge-workspace-backend.cmd
```

---

## License & Attribution

- NVIDIA NIM API integration
- yt-dlp for YouTube processing
- PyMuPDF, python-docx, python-pptx, openpyxl for document processing
- Tesseract OCR for image text extraction

---

**Built with ❤️ using React, FastAPI, and NVIDIA AI.**
