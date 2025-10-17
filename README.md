# AI Wizard API

AI-powered risk analysis wizard for automated asset discovery, impact evaluation, threat identification, and control recommendations.

## Features

- **Multi-step Wizard Flow**: Services → Assets → Impacts → Risks → Controls
- **Web Content Retrieval**: Traditional scraping (BeautifulSoup) or AI-powered web search (OpenAI)
- **Session Management**: Redis-based state persistence across wizard steps
- **LLM-Powered Generation**: Structured output generation using OpenAI models
- **Cost Tracking**: Token usage and cost calculation per request
- **Multi-language Support**: English, Slovak, Czech

## Requirements

- Python 3.11+
- Redis (for session management)
- OpenAI API key

## Installation

### 1. Clone and Setup Virtual Environment

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
# ==================== App Configuration ====================
WORKSHOPS_AI_ENV=production
WORKSHOPS_AI_LOG_LEVEL=INFO
WORKSHOPS_AI_APP_HOST=0.0.0.0
WORKSHOPS_AI_APP_PORT=8000
WORKSHOPS_AI_ENABLE_DOCS=false

# CORS
WORKSHOPS_AI_CORS_ORIGINS=["https://yourdomain.com"]

# ==================== Security ====================
# API Key Authentication
WORKSHOPS_AI_SECURITY__ENABLE_API_KEY=true
WORKSHOPS_AI_SECURITY__API_KEYS=["your-secure-api-key-here"]

# JWT Authentication (optional)
WORKSHOPS_AI_SECURITY__ENABLE_JWT=false

# ==================== Redis ====================
WORKSHOPS_AI_REDIS__URL=redis://localhost:6379/0

# ==================== OpenAI ====================
OPENAI_API_KEY=sk-your-openai-api-key-here
WORKSHOPS_AI_LLM__MODEL=gpt-4o-mini
WORKSHOPS_AI_LLM__TEMPERATURE=0.2

# ==================== Wizard Settings ====================
# Session
WORKSHOPS_AI_WIZARD__SESSION_TTL_HOURS=8

# Web Search (AI-powered)
WORKSHOPS_AI_WIZARD__WEB_SEARCH_ENABLED=true
WORKSHOPS_AI_WIZARD__WEB_SEARCH_MODEL=gpt-4o
WORKSHOPS_AI_WIZARD__WEB_SEARCH_TIMEOUT_SEC=120

# Traditional Scraping
WORKSHOPS_AI_WIZARD__SCRAPE_ENABLED=true
WORKSHOPS_AI_WIZARD__SCRAPE_TIMEOUT_SEC=10
WORKSHOPS_AI_WIZARD__SCRAPE_MAX_RETRIES=2
```

## API Endpoints

### Wizard Flow

All wizard endpoints require authentication via `X-API-Key` header or JWT Bearer token.

#### 1. Scrape Website

```http
POST /scrape/website
Content-Type: application/json
X-API-Key: your-api-key

{
  "url": "https://example.com",
  "search_method": "scrape"  // "scrape" or "web_search"
}
```

**Response:**
```json
{
  "url": "https://example.com",
  "content": "Combined scraped content from homepage, services, and about pages..."
}
```

#### 2. Generate Services

```http
POST /wizard/services
Content-Type: application/json
X-API-Key: your-api-key

{
  "website_url": "https://example.com",
  "search_method": "web_search",
  "language": "en",
  "number_of_services": 3
}
```

**Response:**
```json
{
  "services": [
    {
      "name": "Cloud Hosting",
      "description": "Enterprise cloud infrastructure services",
      "sector": "Digital infrastructure"
    }
  ]
}
```

#### 3. Generate Assets

```http
POST /wizard/assets
Content-Type: application/json
X-API-Key: your-api-key

{
  "language": "en",
  "number_of_assets": 10
}
```

Assets are generated based on services and website content from session.

#### 4. Generate Impacts

```http
POST /wizard/impacts
Content-Type: application/json
X-API-Key: your-api-key

{
  "language": "en"
}
```

#### 5. Generate Risks

```http
POST /wizard/risks
Content-Type: application/json
X-API-Key: your-api-key

{
  "language": "en"
}
```

#### 6. Generate Controls

```http
POST /wizard/controls
Content-Type: application/json
X-API-Key: your-api-key

{
  "language": "en",
  "max_controls_per_asset": 5
}
```

#### 7. Full Evaluation (All Steps)

```http
POST /wizard/evaluation
Content-Type: application/json
X-API-Key: your-api-key

{
  "website_url": "https://example.com",
  "search_method": "web_search",
  "language": "en",
  "number_of_services": 3,
  "number_of_assets": 10,
  "max_controls_per_asset": 5
}
```

Runs all wizard steps in sequence and returns combined results.

### State Management

```http
# Get current wizard state
GET /wizard/state?org_id=org123&user_id=user456

# Clear wizard state (reset)
DELETE /wizard/state?org_id=org123&user_id=user456
```

## Configuration

### Session Management

- **TTL**: Configurable via `WORKSHOPS_AI_WIZARD__SESSION_TTL_HOURS` (default: 8 hours)
- **Storage**: Redis key pattern: `wizard:{org_id}:{user_id}`

### Web Scraping

Two methods available:

1. **Traditional Scraping** (`search_method="scrape"`):
   - BeautifulSoup-based HTML parsing
   - Fetches homepage + services + about pages
   - Content sanitization and validation
   - Prompt injection protection

2. **AI Web Search** (`search_method="web_search"`):
   - OpenAI-powered intelligent search
   - Domain-filtered results
   - Higher quality content extraction

### Security

- **API Key Auth**: Header `X-API-Key: your-key` or query param `?api_key=your-key`
- **JWT Auth**: Header `Authorization: Bearer <token>`
- **Content Sanitization**: Removes suspicious patterns and potential prompt injections

## Cost Tracking

All LLM operations return cost information:

```json
{
  "debug": {
    "usage": {
      "prompt_tokens": 1234,
      "completion_tokens": 567,
      "total_tokens": 1801,
      "cost_usd": 0.00123,
      "model": "gpt-4o-mini"
    }
  }
}
```

Enable debug info via `WORKSHOPS_AI_WIZARD__EXPOSE_DEBUG_INFO=true`.

## Architecture

### Components

- **FastAPI**: REST API framework
- **Redis**: Session storage and caching
- **OpenAI**: LLM for generation and web search
- **BeautifulSoup + lxml**: HTML parsing for traditional scraping
- **Pydantic**: Request/response validation
- **Tenacity**: Retry logic for transient failures

### Scraper Architecture

```
workshops_ai/wizard/scraper/
├── core.py              # Main get_website() orchestration
├── http_client.py       # Shared HTTP client with connection reuse
├── text_processing.py   # Unicode normalization, encoding fixes
├── sanitizer.py         # Security: prompt injection detection
├── validation.py        # Content quality checks
├── link_extractor.py    # Same-domain link extraction (PSL-based)
└── constants.py         # URL patterns for services/about pages
```

### Session Flow

1. **Scrape Website** → Content cached in Redis (`website_content`)
2. **Generate Services** → Reads cached content, generates services
3. **Generate Assets** → Reads services from session, generates assets
4. **Generate Impacts** → Reads assets from session
5. **Generate Risks** → Reads assets + impacts
6. **Generate Controls** → Reads assets + risks

Each step saves results to session for use by subsequent steps.