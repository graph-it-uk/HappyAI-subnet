from dotenv import load_dotenv
import os
from supabase import create_client


load_dotenv()
supabase_mode = os.environ.get("SUPABASE_MODE", "False").lower() == "true"


if supabase_mode:
    supabase = create_client(
        supabase_url=os.environ.get("SUPABASE_URL"),
        supabase_key=os.environ.get("SUPABASE_KEY")
    )

if supabase_mode:
    result = supabase.table('evaluations').insert({
        "user_id": "1",
        "query": "What is the capital of France111?",
        "response": "Paris",
        "reference_result": "Paris",
        "score_semantic": 1,
        "score_judge": 1,
        "combo_score": 1
    }).execute()
    
    print(result)

