POST /ux-agent

Input:
{
  "user_request": "Build me a dashboard for FinPro"
}

State:
{
  "user_request": string,
  "company_profile": object | null,
  "product_profile": object | null,
  "context_analysis": object | null,
  "questions": list,
  "user_answers": list,
  "final_recommendation": object | null
}

user prompt request 
  - user_id
  - profile 