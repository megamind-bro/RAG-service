import os
import json
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import orjson
from pypdf import PdfReader

from .retriever import Retriever
from .generator import Generator
from .database import ConversationDatabase
from .analytics import analytics
from .schemas import (
    IngestRequest, 
    QueryRequest, 
    QueryResponse, 
    CreateConversationRequest,
    UpdateConversationRequest,
    HealthResponse,
    AdviceRequest,
    AdviceResponse
)
from .models import Conversation, ConversationSummary
import logging

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
INDEX_DIR = os.getenv("INDEX_DIR", "data")
CONVERSATION_DIR = os.getenv("CONVERSATION_DIR", "data/conversations")
PORT = int(os.getenv("PORT", "8088"))
ADVICE_DEBUG = os.getenv("ADVICE_DEBUG", "false").lower() == "true"

app = FastAPI(
    title="Grocery Store RAG Service",
    description="AI-powered grocery store assistant with product recommendations, recipes, and inventory management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
retriever = Retriever(embedding_model_name=EMBEDDING_MODEL, index_dir=INDEX_DIR)
generator = Generator(model_name=GENERATION_MODEL)
conversation_db = ConversationDatabase(db_dir=CONVERSATION_DIR)

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat(),
        services={
            "retriever": "operational",
            "generator": "operational",
            "database": "operational"
        }
    )

@app.post("/ingest", tags=["Document Management"])
def ingest(req: IngestRequest):
    """
    Ingest documents into the knowledge base
    
    - **documents**: List of documents with text and optional metadata
    """
    retriever.add_documents([d.model_dump() for d in req.documents])
    retriever.persist()
    return {
        "status": "success",
        "message": "Documents ingested successfully",
        "count": len(req.documents)
    }

@app.post("/ingest/files", tags=["Document Management"])
async def ingest_files(files: List[UploadFile] = File(...)):
    """
    Upload and ingest files (PDF or text) into the knowledge base
    
    - **files**: List of files to upload (supports PDF and text files)
    """
    docs: List[Dict[str, Any]] = []
    for f in files:
        raw = await f.read()
        name = (f.filename or "").lower()
        if name.endswith(".pdf"):
            # Parse PDF bytes
            from io import BytesIO
            reader = PdfReader(BytesIO(raw))
            pages_text: List[str] = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            text = "\n\n".join(pages_text)
            docs.append({
                "id": f.filename, 
                "text": text, 
                "metadata": {"filename": f.filename, "type": "pdf"}
            })
        else:
            text = raw.decode("utf-8", errors="ignore")
            docs.append({
                "id": f.filename, 
                "text": text, 
                "metadata": {"filename": f.filename, "type": "text"}
            })
    
    retriever.add_documents(docs)
    retriever.persist()
    return {
        "status": "success",
        "message": "Files ingested successfully",
        "count": len(docs),
        "files": [f.filename for f in files]
    }

@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query(req: QueryRequest):
    """
    Query the RAG system with conversation history support
    
    - **question**: The question to ask
    - **k**: Number of context documents to retrieve (1-10)
    - **conversation_id**: Optional conversation ID for history tracking
    - **user_id**: User identifier (required)
    """
    # Start analytics tracking
    query_id = str(uuid.uuid4())
    tracking_data = analytics.start_query_tracking(
        query_id=query_id,
        user_id=req.user_id,
        query_text=req.question,
        endpoint="/query"
    )
    
    try:
        # Retrieve relevant contexts
        analytics.mark_retrieval_start(tracking_data)
        results = retriever.search(req.question, k=req.k)
        
        # Get conversation history for context
        conversation_history = []
        if req.conversation_id:
            existing_conversation = conversation_db.get_conversation(req.conversation_id)
            if existing_conversation:
                # Convert messages to dict format for generator
                conversation_history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in existing_conversation.messages
                ]
        
        # Generate answer with conversation context
        analytics.mark_generation_start(tracking_data)
        answer = generator.generate(
            question=req.question,
            contexts=[r["text"] for r in results],
            conversation_history=conversation_history
        )
        
        # Handle conversation history
        conversation_id = req.conversation_id
        if not conversation_id:
            # Create new conversation if not provided
            conversation = conversation_db.create_conversation(
                user_id=req.user_id,
                title=req.question[:50] + ("..." if len(req.question) > 50 else "")
            )
            conversation_id = conversation.conversation_id
        
        # Add user message
        conversation_db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=req.question
        )
        
        # Add assistant message with contexts
        conversation_db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            contexts=results
        )
        
        # Complete analytics tracking
        analytics.complete_query_tracking(
            tracking_data=tracking_data,
            retrieved_contexts=results,
            response=answer,
            success=True,
            conversation_id=conversation_id
        )
        
        return QueryResponse(
            answer=answer,
            contexts=results,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        # Track failed query
        analytics.complete_query_tracking(
            tracking_data=tracking_data,
            retrieved_contexts=[],
            response="",
            success=False,
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


def _format_grocery_context(ctx: dict) -> str:
    # Create a compact, readable context block for grocery store prompting
    lines = []
    lines.append(f"Store Location: {ctx.get('store_location', 'Main Store')}")
    lines.append(f"Customer Type: {ctx.get('customer_type', 'Regular')}")
    
    if ctx.get("budget"):
        lines.append(f"Budget: ${ctx.get('budget')}")
    if ctx.get("dietary_preferences"):
        lines.append(f"Dietary Preferences: {', '.join(ctx.get('dietary_preferences', []))}")
    if ctx.get("allergies"):
        lines.append(f"Allergies: {', '.join(ctx.get('allergies', []))}")
    if ctx.get("favorite_products"):
        lines.append(f"Favorite Products: {', '.join(ctx.get('favorite_products', []))}")
    if ctx.get("shopping_frequency"):
        lines.append(f"Shopping Frequency: {ctx.get('shopping_frequency')}")
    
    if ctx.get("store_inventory"):
        lines.append("Available Products:")
        for item in ctx.get("store_inventory", [])[:10]:
            lines.append(f"  - {item.get('name')} (${item.get('price')}, Stock: {item.get('quantity')})")
    
    if ctx.get("current_promotions"):
        lines.append("Current Promotions:")
        for promo in ctx.get("current_promotions", []):
            lines.append(f"  - {promo}")
    
    return "\n".join(lines)


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_list(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, list):
        return [str(v).strip() for v in values if str(v).strip()]
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    return []


def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _decorate_shopping_advice(text: str) -> str:
    base = (text or "").strip()
    if not base:
        base = "Here are personalized shopping recommendations for you."
    greeting = "Hello, valued customer!"
    lower = base.lower()
    if not lower.startswith(("hello", "hi ", "hi,", "dear", "greetings", "welcome", "good morning", "good afternoon", "good evening")):
        base = f"{greeting}\n\n{base}"
    closing = "\n\nHappy shopping!\nGrocery Store Assistant"
    if "grocery store assistant" not in base.lower():
        base = base.rstrip() + closing
    return base


def _extract_grocery_recommendations(answer: str) -> Dict[str, List[str]]:
    sections = {
        "products": [],
        "recipes": [],
        "deals": [],
        "health_tips": []
    }
    if not answer:
        return sections

    current = None
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        normalized = line.lower().rstrip(":")
        if "product" in normalized or "buy" in normalized:
            current = "products"
            continue
        if "recipe" in normalized or "dish" in normalized or "meal" in normalized:
            current = "recipes"
            continue
        if "deal" in normalized or "sale" in normalized or "discount" in normalized or "promo" in normalized:
            current = "deals"
            continue
        if "health" in normalized or "nutrition" in normalized or "tip" in normalized:
            current = "health_tips"
            continue

        bullet = line[0] in {"-", "*", "•"} or (line[:2].isdigit() and line[2:3] in {".", ")", ":"})
        if bullet:
            cleaned = line.lstrip("-*• \t0123456789.).").strip()
            if not cleaned:
                continue
            target = current or "products"
            sections[target].append(cleaned)
        elif current and len(line.split()) <= 14:
            sections[current].append(line)

    return sections


def _structured_lists_empty(lists: Dict[str, List[str]]) -> bool:
    return not any(lists.values())


def _generate_structured_lists_with_model(ctx: Dict[str, Any], narrative: str) -> Dict[str, List[str]]:
    prompt = (
        "You are an agricultural advisor. Convert the analytics summary and narrative below into structured bullet lists. "
        "Return STRICT JSON with keys fertilizer_recommendations, prioritized_actions, risk_warnings, seed_recommendations. "
        "Each value must be a list of strings with rich, specific guidance.\n\n"
        f"Analytics Summary:\n{_format_analytics_context(ctx)}\n\n"
        f"Narrative Advice:\n{narrative}\n"
    )
    follow_up = generator.generate(prompt, [])
    try:
        start = follow_up.find("{")
        end = follow_up.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        parsed = json.loads(follow_up[start:end + 1])
        return {
            "fertilizer": _clean_list(parsed.get("fertilizer_recommendations")),
            "actions": _clean_list(parsed.get("prioritized_actions")),
            "risks": _clean_list(parsed.get("risk_warnings")),
            "seeds": _clean_list(parsed.get("seed_recommendations"))
        }
    except Exception:
        return {}


def _create_grocery_recommendation_prompt(ctx: Dict[str, Any]) -> str:
    """Create a context-aware prompt for grocery store recommendations"""
    customer_type = ctx.get("customer_type", "Regular Customer")
    store_location = ctx.get("store_location", "Main Store")
    
    # Analyze customer preferences to tailor recommendations
    dietary_prefs = ctx.get("dietary_preferences", [])
    allergies = ctx.get("allergies", [])
    budget = _to_float(ctx.get("budget"))
    shopping_freq = ctx.get("shopping_frequency", "Weekly")
    
    # Determine focus areas based on customer profile
    focus_areas = []
    if dietary_prefs:
        focus_areas.append(f"dietary requirements ({', '.join(dietary_prefs)})")
    if allergies:
        focus_areas.append(f"allergies ({', '.join(allergies)})")
    if budget and budget < 50:
        focus_areas.append("budget-friendly options")
    if shopping_freq == "Monthly":
        focus_areas.append("long-shelf-life products and bulk deals")
    
    focus_text = f"Pay special attention to: {', '.join(focus_areas)}." if focus_areas else ""
    
    prompt = f"""As a helpful grocery store assistant, analyze the customer profile and provide personalized shopping recommendations.

ANALYSIS REQUIREMENTS:
1. Understand customer dietary preferences and allergies
2. Suggest products that match their lifestyle and budget
3. Recommend recipes using available in-store products
4. Highlight current promotions and deals
5. {focus_text}

RESPONSE FORMAT:
Return your response as valid JSON with these exact keys:
{{
    "recommendation": "Personalized shopping advice (200-300 words) addressing the customer directly",
    "product_suggestions": ["Specific product recommendations with benefits"],
    "recipe_ideas": ["Easy recipes using available products"],
    "current_deals": ["Relevant promotions and sales"],
    "health_tips": ["Nutrition and shopping tips"]
}}

GUIDELINES:
- Consider budget constraints
- Avoid allergenic ingredients
- Focus on healthier options when possible
- Provide actionable suggestions
- Include product names and departments when available
- Highlight value and savings opportunities

Generate recommendations that will help this {customer_type} make the most of their shopping experience at {store_location}."""

    return prompt


def _create_grocery_fallback_advice(ctx: Dict[str, Any], fallback_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Create enhanced fallback advice for grocery shopping when AI generation fails"""
    customer_type = ctx.get("customer_type", "Customer")
    store_location = ctx.get("store_location", "our store")
    budget = _to_float(ctx.get("budget"))
    dietary_prefs = ctx.get("dietary_preferences", [])
    
    # Create personalized advice based on customer profile
    advice_parts = [
        f"Welcome to {store_location}! Based on your shopping preferences, I've curated personalized recommendations for your visit."
    ]
    
    if dietary_prefs:
        diet_list = ", ".join(dietary_prefs)
        advice_parts.append(f"We've identified your dietary preferences ({diet_list}) and highlighted matching products throughout the store.")
    
    if budget is not None:
        if budget < 30:
            advice_parts.append(f"With a budget of ${budget}, we've focused on our best value items and current promotions to maximize your savings.")
        elif budget < 75:
            advice_parts.append(f"With your budget of ${budget}, we have excellent options across all departments with great quality-to-price ratios.")
        else:
            advice_parts.append(f"With your ${budget} budget, you can explore our premium and specialty selections for a gourmet experience.")
    
    if ctx.get("allergies"):
        allergies_list = ", ".join(ctx.get("allergies", []))
        advice_parts.append(f"⚠️ We've marked all items containing {allergies_list} for your safety. Always check labels before purchase.")
    
    advice_parts.append("The recommendations below are tailored to your preferences and store availability.")
    
    enhanced_advice = _decorate_shopping_advice(" ".join(advice_parts))
    
    return {
        "recommendation": enhanced_advice,
        "product_suggestions": fallback_plan.get("product_suggestions", []),
        "recipe_ideas": fallback_plan.get("recipe_ideas", []),
        "current_deals": fallback_plan.get("current_deals", []),
        "health_tips": fallback_plan.get("health_tips", [])
    }


def _generate_grocery_recommendations(ctx: Dict[str, Any]) -> Dict[str, Any]:
    customer_type = ctx.get("customer_type") or "Regular Customer"
    store_location = ctx.get("store_location") or "our store"
    advice_sections: List[str] = []
    product_suggestions: List[str] = []
    recipe_ideas: List[str] = []
    current_deals: List[str] = []
    health_tips: List[str] = []

    budget = _to_float(ctx.get("budget"))
    dietary_prefs = ctx.get("dietary_preferences", [])
    allergies = ctx.get("allergies", [])
    favorite_products = ctx.get("favorite_products", [])
    shopping_freq = ctx.get("shopping_frequency", "Weekly")
    store_inventory = ctx.get("store_inventory", [])
    promos = ctx.get("current_promotions", [])

    # Build personalized greeting
    summary_parts: List[str] = []
    summary_parts.append(f"Welcome to {store_location}!")
    if budget:
        summary_parts.append(f"With your budget of ${budget:.2f}")
    if shopping_freq:
        summary_parts.append(f"for your {shopping_freq.lower()} shopping")
    
    if summary_parts:
        advice_sections.append(" ".join(summary_parts) + ", here are our top recommendations.")

    # Dietary and allergy preferences
    if dietary_prefs:
        diet_text = f"We've identified your dietary preferences: {', '.join(dietary_prefs)}. "
        diet_text += "Look for our special shelf markers highlighting compatible products in each department."
        advice_sections.append(diet_text)
    
    if allergies:
        allergy_text = f"⚠️ Please be aware of these allergies in your household: {', '.join(allergies)}. "
        allergy_text += "Always check product labels, and ask our staff if you have questions."
        advice_sections.append(allergy_text)

    # Budget-based suggestions
    if budget is not None:
        if budget < 40:
            advice_sections.append("We've focused on our everyday value brands and current promotions to stretch your budget.")
            product_suggestions.append("Store-brand staples offer excellent value without sacrificing quality")
            product_suggestions.append("Check the clearance sections for seasonal and deep-discount items")
        elif budget < 100:
            advice_sections.append("Your budget allows for a good mix of quality brands and premium selections.")
            product_suggestions.append("Mid-range organic options provide health benefits at reasonable prices")
        else:
            advice_sections.append("Your budget allows us to recommend premium products and specialty items.")
            product_suggestions.append("Explore our gourmet and specialty sections for high-quality, artisanal products")

    # Product-specific guidance
    if store_inventory:
        advice_sections.append(f"We have {len(store_inventory)} fresh products in stock today.")
        for item in store_inventory[:5]:
            if item.get('quantity', 0) > 0:
                product_suggestions.append(f"{item.get('name')} - ${item.get('price')} (in stock)")

    # Deals and promotions
    if promos:
        advice_sections.append("Don't miss our current promotions:")
        for promo in promos[:3]:
            current_deals.append(promo)
    else:
        current_deals.append("Check our loyalty app for exclusive member-only deals")

    # Recipe ideas based on dietary preferences
    if "vegan" in [p.lower() for p in dietary_prefs]:
        recipe_ideas.append("Plant-based Buddha bowls with seasonal vegetables")
        recipe_ideas.append("Chickpea curry with coconut milk and fresh herbs")
        recipe_ideas.append("Smoothie bowls with almond butter and fresh berries")
    elif "keto" in [p.lower() for p in dietary_prefs]:
        recipe_ideas.append("Grilled salmon with lemon butter and asparagus")
        recipe_ideas.append("Cauliflower rice stir-fry with grass-fed beef")
        recipe_ideas.append("Zucchini noodles with pesto and pine nuts")
    else:
        recipe_ideas.append("Sheet pan roasted chicken with seasonal vegetables")
        recipe_ideas.append("Fresh salad bowls with grilled proteins")
        recipe_ideas.append("Pasta dishes with homemade tomato sauce")

    # Health tips
    health_tips.append("Buy fresh produce at the peak of the season for maximum nutrition and best prices")
    health_tips.append("Read nutrition labels to compare sodium, sugar, and added ingredients")
    if "organic" in [p.lower() for p in dietary_prefs]:
        health_tips.append("Prioritize organic versions of items on the 'Dirty Dozen' list")
    health_tips.append("Stock your pantry with fiber-rich whole grains and legumes")

    # Shopping tips by frequency
    if shopping_freq == "Monthly":
        advice_sections.append("For your monthly shopping, focus on non-perishables and freeze fresh items for later use.")
        product_suggestions.append("Bulk frozen vegetables maintain nutrition and reduce waste")
        product_suggestions.append("Long-shelf-life staples like pasta, rice, and canned beans")

    if not product_suggestions:
        product_suggestions.append("Fresh seasonal produce is currently available in our produce section")
    if not recipe_ideas:
        recipe_ideas.append("Ask our staff for recipe cards featuring store products")
    if not current_deals:
        current_deals.append("Visit customer service for a list of this week's promotions")
    if not health_tips:
        health_tips.append("Our nutritionist is available for free consultations during store hours")

    product_suggestions = _dedupe(product_suggestions)
    recipe_ideas = _dedupe(recipe_ideas)
    current_deals = _dedupe(current_deals)
    health_tips = _dedupe(health_tips)

    advice_text = " ".join(advice_sections) if advice_sections else f"Welcome to {store_location}! We're here to help you find everything you need."
    return {
        "recommendation": advice_text,
        "product_suggestions": product_suggestions,
        "recipe_ideas": recipe_ideas,
        "current_deals": current_deals,
        "health_tips": health_tips
    }

    summary_parts: List[str] = []
    if total_revenue is not None:
        summary_parts.append(f"generated revenue of approximately {total_revenue:.2f}")
    if total_expenses is not None:
        summary_parts.append(f"spent about {total_expenses:.2f} in inputs")
    if net_profit is not None:
        summary_parts.append(f"resulted in net profit near {net_profit:.2f}")
    if summary_parts:
        advice_sections.append(
            f"In the recent period, your {crop} enterprise {' and '.join(summary_parts)}."
        )

    if avg_yield is not None and best_yield is not None:
        advice_sections.append(
            f"Average yield per unit sits at {avg_yield:.2f}, with best-performing plots reaching {best_yield:.2f}."
        )
    elif avg_yield is not None:
        advice_sections.append(f"Average yield per unit is {avg_yield:.2f}.")

    if profit_margin is not None:
        if profit_margin < 15:
            advice_sections.append("Profit margins are tight; focus on reducing input costs and improving yield efficiency.")
            prioritized_actions.append("Review major expense categories and negotiate bulk pricing for inputs.")
        elif profit_margin < 30:
            advice_sections.append("Margins are moderate; optimize fertilization and pest control to raise profitability.")
        else:
            advice_sections.append("Profit margins look strong—maintain cost discipline while scaling the practices that deliver the best results.")

    if net_profit is not None and net_profit < 0:
        advice_sections.append("Net profit is negative; prioritize high-impact interventions and pause low-return activities.")
        risk_warnings.append("Current season is operating at a loss; reassess selling prices and input expenses urgently.")

    if avg_yield and best_yield and best_yield - avg_yield > max(0.2 * avg_yield, 0.1):
        advice_sections.append("Top-performing fields are far outpacing the average. Audit their management and replicate those practices on weaker plots.")
        prioritized_actions.append("Document the management of your highest-yielding field and apply the same schedule to underperforming areas.")

    if worst_yield and avg_yield and worst_yield < 0.8 * avg_yield:
        risk_warnings.append("Some plots are well below average yield; investigate soil fertility, pest pressure, or water stress.")

    fertilizer_recommendations.extend(_fertilizer_guidance(crop, soil_ph))
    seed_recommendations.extend(_seed_guidance(crop, location, rainfall))

    pesticide_cost = expenses_by_category.get("PESTICIDES")
    fertilizer_cost = expenses_by_category.get("FERTILIZER")
    if pesticide_cost and fertilizer_cost:
        try:
            pesticide_cost = float(pesticide_cost)
            fertilizer_cost = float(fertilizer_cost)
            if pesticide_cost > fertilizer_cost * 0.5:
                advice_sections.append("Pesticide spending is high. Deploy integrated pest management to cut chemical dependence.")
                prioritized_actions.append("Scout fields weekly and use targeted biological controls to lower pesticide costs.")
        except (TypeError, ValueError):
            pass

    if rainfall is not None:
        if rainfall < 350:
            advice_sections.append("Rainfall is below ideal levels; concentrate on moisture conservation and supplemental irrigation.")
            prioritized_actions.append("Apply mulch and schedule irrigation around critical growth stages.")
            risk_warnings.append("Low rainfall increases drought stress risk; monitor soil moisture closely.")
        elif rainfall < 500:
            advice_sections.append("Rainfall is moderately low; focus on drought-tolerant varieties and timely weeding to reduce moisture competition.")
        elif rainfall > 800:
            advice_sections.append("High rainfall can lead to nutrient leaching; adjust fertilization to compensate for losses.")
            risk_warnings.append("Excess rainfall raises disease risk; tighten your crop protection schedule.")
            prioritized_actions.append("Improve drainage around low-lying fields to avoid waterlogging.")

    if dominant_weather:
        if "dry" in dominant_weather or "drought" in dominant_weather:
            advice_sections.append("Weather outlook indicates dry spells; schedule irrigation shifts early mornings and late evenings.")
            risk_warnings.append("Extended dry periods increase risk of silk desiccation; maintain soil moisture during tasseling.")
            prioritized_actions.append("Install soil moisture sensors or manual probes to trigger irrigation before wilting.")
        if "wet" in dominant_weather or "rain" in dominant_weather:
            advice_sections.append("Expect wetter conditions; scout for fungal diseases weekly and rotate fungicides to prevent resistance.")
            risk_warnings.append("Persistent moisture favours leaf blights and rust; keep protective sprays ready.")
        if "wind" in dominant_weather or "storm" in dominant_weather:
            risk_warnings.append("High winds can cause lodging; consider staking or strategic windbreaks near exposed fields.")

    if total_expenses and total_revenue and total_expenses > total_revenue:
        risk_warnings.append("Expenses exceed revenue for this period; prioritize profitable crops and defer non-essential purchases.")

    if not prioritized_actions:
        prioritized_actions.append("Keep detailed records of field activities and yields to fine-tune next season’s plan.")

    if not fertilizer_recommendations:
        fertilizer_recommendations.append("Apply a balanced N-P-K fertilizer at planting and supplement with nitrogen at knee-high stage.")
    if not seed_recommendations:
        seed_recommendations.append(f"Select certified, drought-resilient hybrid seed suitable for {crop}, such as H516 or DK8031.")
    prioritized_actions.append("Harvest at physiological maturity and dry maize to 13% moisture before storage.")

    storage_tip = "Dry shelled maize thoroughly and store in airtight hermetic bags or silos to prevent aflatoxin buildup."
    if storage_tip not in risk_warnings:
        risk_warnings.append(storage_tip)

    fertilizer_recommendations = _dedupe(fertilizer_recommendations)
    prioritized_actions = _dedupe(prioritized_actions)
    risk_warnings = _dedupe(risk_warnings)
    seed_recommendations = _dedupe(seed_recommendations)

    advice_text = " ".join(advice_sections) if advice_sections else f"Continue refining your {crop} management with careful record keeping and timely field operations."
    return {
        "advice": advice_text,
        "fertilizer_recommendations": fertilizer_recommendations,
        "prioritized_actions": prioritized_actions,
        "risk_warnings": risk_warnings if risk_warnings else ["Continue monitoring crop health and market prices each week."],
        "seed_recommendations": seed_recommendations
    }


def _product_category_guidance(dietary_prefs: List[str]) -> List[str]:
    recs: List[str] = []
    if not dietary_prefs:
        recs.append("Explore all departments for a balanced variety of proteins, grains, and fresh produce")
        return recs
    
    for pref in dietary_prefs:
        pref_lower = pref.lower()
        if "vegan" in pref_lower:
            recs.append("Visit our expanded plant-based section for meat alternatives, legumes, and dairy-free products")
        elif "vegetarian" in pref_lower:
            recs.append("Our dairy and egg sections are fully stocked with vegetarian-friendly products")
        elif "keto" in pref_lower or "low-carb" in pref_lower:
            recs.append("Focus on meats, fish, eggs, and low-carb vegetables in produce and frozen sections")
        elif "gluten" in pref_lower or "celiac" in pref_lower:
            recs.append("Check our dedicated gluten-free section for certified products and alternatives")
        elif "organic" in pref_lower:
            recs.append("Our organic section includes certified produce, meat, and pantry staples")
        else:
            recs.append(f"We have many products suitable for your {pref} dietary preference")
    return recs


def _seed_guidance(crop: str, location: Optional[str], rainfall: Optional[float]) -> List[str]:
    crop_name = crop or "your crop"
    regional_hint = f" for {location}" if location else ""
    recs: List[str] = []
    if rainfall is not None and rainfall < 400:
        recs.append(f"Choose early-maturing, drought-tolerant hybrids{regional_hint} such as DUMA 43, DK8031, or PIONEER P2859.")
    elif rainfall is not None and rainfall > 700:
        recs.append(f"Opt for high-yield, disease-tolerant hybrids{regional_hint} like H629 or PIONEER P3812 that perform well in wetter zones.")
    else:
        recs.append(f"Use certified hybrid seed{regional_hint}, e.g., H614D or SC DUMA 43, and treat seed with a systemic fungicide/insecticide before planting.")
    return recs


@app.post("/advice", response_model=AdviceResponse, tags=["Advisory"])
def advice(req: AdviceRequest):
    """
    Generate prescriptive advice and fertilizer recommendations using provided analytics context.
    """
    # Start analytics tracking
    query_id = str(uuid.uuid4())
    tracking_data = analytics.start_query_tracking(
        query_id=query_id,
        user_id=req.user_id,
        query_text=f"Advice request for {req.context.crop_type}",
        endpoint="/advice"
    )
    
    try:
        ctx_dict = req.context.model_dump()
        analytics_block = _format_analytics_context(ctx_dict)

        # Create enhanced prompt for maize-specific advice
        question = _create_enhanced_advice_prompt(ctx_dict)

        # Retrieve context from knowledge base
        analytics.mark_retrieval_start(tracking_data)
        results = retriever.search(
            f"Best agronomic practices and fertilizer programs for {req.context.crop_type}. Soil={req.context.soil_type} pH={req.context.soil_ph}",
            k=req.k
        )

        # Build a single prompt including analytics
        prompt = (
            "You are an agricultural advisor for smallholder and commercial farmers.\n\n"
            "Farm Analytics:\n"
            f"{analytics_block}\n\n"
            "Task:\n"
            f"{question}\n"
            "Ensure JSON is valid and does not include markdown code fences."
        )

        analytics.mark_generation_start(tracking_data)
        # Try AI generation first, fallback to enhanced rule-based system
        print(f"🌾 Generating intelligent advice for {ctx_dict.get('crop_type', 'maize')} farmer in {ctx_dict.get('location', 'Kenya')}")
        
        try:
            # Attempt AI-powered advice generation
            answer = generator.generate_advice(
                analytics_context=analytics_block,
                enhanced_prompt=question,
                contexts=[r["text"] for r in results]
            )
            
            # Try to parse AI response
            import json
            try:
                ai_advice = json.loads(answer)
                print("✅ AI-powered advice generated successfully")
                
                # Complete analytics tracking for AI advice
                analytics.complete_query_tracking(
                    tracking_data=tracking_data,
                    retrieved_contexts=results,
                    response=ai_advice.get("advice", "AI-generated advice"),
                    success=True
                )
                
                return AdviceResponse(
                    advice=ai_advice.get("advice", "AI-generated agricultural advice"),
                    fertilizer_recommendations=ai_advice.get("fertilizer_recommendations", []),
                    prioritized_actions=ai_advice.get("prioritized_actions", []),
                    risk_warnings=ai_advice.get("risk_warnings", []),
                    seed_recommendations=ai_advice.get("seed_recommendations", []),
                    contexts=results
                )
            except json.JSONDecodeError:
                print("⚠️ AI response parsing failed, using enhanced fallback")
                
        except Exception as ai_error:
            print(f"⚠️ AI generation failed: {str(ai_error)}, using enhanced fallback")
        
        # Enhanced rule-based system (current working system)
        fallback_plan = _generate_rule_based_recommendations(ctx_dict)
        enhanced_advice = _create_enhanced_fallback_advice(ctx_dict, fallback_plan)
        print("🧠 Using enhanced rule-based intelligence system")
        
        # Complete analytics tracking for enhanced advice
        analytics.complete_query_tracking(
            tracking_data=tracking_data,
            retrieved_contexts=results,
            response=enhanced_advice["advice"],
            success=True
        )
        
        return AdviceResponse(
            advice=enhanced_advice["advice"],
            fertilizer_recommendations=enhanced_advice["fertilizer_recommendations"],
            prioritized_actions=enhanced_advice["prioritized_actions"],
            risk_warnings=enhanced_advice["risk_warnings"],
            seed_recommendations=enhanced_advice["seed_recommendations"],
            contexts=results
        )
    
    except Exception as e:
        # Track failed advice request
        analytics.complete_query_tracking(
            tracking_data=tracking_data,
            retrieved_contexts=[],
            response="",
            success=False,
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Advice generation failed: {str(e)}")
# ==================== Conversation Management Endpoints ====================

@app.post("/conversations", response_model=Conversation, tags=["Conversations"])
def create_conversation(req: CreateConversationRequest):
    """
    Create a new conversation
    
    - **user_id**: User identifier
    - **title**: Optional conversation title
    - **metadata**: Optional metadata
    """
    conversation = conversation_db.create_conversation(
        user_id=req.user_id,
        title=req.title,
        metadata=req.metadata
    )
    return conversation


@app.get("/conversations/{conversation_id}", response_model=Conversation, tags=["Conversations"])
def get_conversation(conversation_id: str):
    """
    Get a specific conversation by ID
    
    - **conversation_id**: The conversation identifier
    """
    conversation = conversation_db.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/users/{user_id}/conversations", response_model=List[ConversationSummary], tags=["Conversations"])
def get_user_conversations(
    user_id: str,
    limit: int = Query(50, ge=1, le=100, description="Maximum number of conversations to return"),
    offset: int = Query(0, ge=0, description="Number of conversations to skip")
):
    """
    Get all conversations for a user
    
    - **user_id**: User identifier
    - **limit**: Maximum number of conversations to return (1-100)
    - **offset**: Number of conversations to skip for pagination
    """
    conversations = conversation_db.get_user_conversations(
        user_id=user_id,
        limit=limit,
        offset=offset
    )
    return conversations


@app.patch("/conversations/{conversation_id}", response_model=Conversation, tags=["Conversations"])
def update_conversation(conversation_id: str, req: UpdateConversationRequest):
    """
    Update conversation metadata
    
    - **conversation_id**: The conversation identifier
    - **title**: Optional new title
    - **metadata**: Optional new metadata
    """
    conversation = conversation_db.update_conversation(
        conversation_id=conversation_id,
        title=req.title,
        metadata=req.metadata
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{conversation_id}", tags=["Conversations"])
def delete_conversation(conversation_id: str, user_id: str = Query(..., description="User ID for authorization")):
    """
    Delete a conversation
    
    - **conversation_id**: The conversation identifier
    - **user_id**: User identifier (for authorization)
    """
    success = conversation_db.delete_conversation(conversation_id, user_id)
    if not success:
        raise HTTPException(
            status_code=404, 
            detail="Conversation not found or unauthorized"
        )
    return {"status": "success", "message": "Conversation deleted successfully"}


# ==================== Analytics Endpoints ====================

@app.get("/analytics/performance", tags=["Analytics"])
def get_performance_analytics(days: int = Query(7, ge=1, le=90, description="Number of days to analyze")):
    """
    Get system performance analytics for the specified period
    
    - **days**: Number of days to analyze (1-90)
    """
    try:
        insights = analytics.get_performance_insights(days=days)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance analytics: {str(e)}")


@app.get("/analytics/users/{user_id}", tags=["Analytics"])
def get_user_analytics(
    user_id: str, 
    days: int = Query(30, ge=1, le=90, description="Number of days to analyze")
):
    """
    Get analytics for a specific user
    
    - **user_id**: User identifier
    - **days**: Number of days to analyze (1-90)
    """
    try:
        user_data = analytics.get_user_analytics(user_id=user_id, days=days)
        return user_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user analytics: {str(e)}")


@app.post("/analytics/feedback", tags=["Analytics"])
def record_feedback(
    query_id: str = Query(..., description="Query ID to provide feedback for"),
    rating: int = Query(..., ge=1, le=5, description="Rating from 1-5")
):
    """
    Record user feedback for a query
    
    - **query_id**: The query ID to provide feedback for
    - **rating**: Rating from 1 (poor) to 5 (excellent)
    """
    try:
        analytics.record_user_feedback(query_id=query_id, rating=rating)
        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@app.get("/analytics/report", tags=["Analytics"])
def export_analytics_report(days: int = Query(30, ge=1, le=90, description="Number of days to include in report")):
    """
    Export comprehensive analytics report
    
    - **days**: Number of days to include in the report (1-90)
    """
    try:
        report = analytics.export_analytics_report(days=days)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate analytics report: {str(e)}") 