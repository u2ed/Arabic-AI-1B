const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type"
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
    if (request.method !== "POST") return Response.json({ error: "Method not allowed" }, { status: 405, headers: corsHeaders });
    try {
      const body = await request.json();
      const messages = Array.isArray(body.messages) ? body.messages : [];
      if (!messages.length) return Response.json({ error: "messages is required" }, { status: 400, headers: corsHeaders });
      const result = await env.AI.run("@cf/meta/llama-3.1-8b-instruct", {
        messages: [
          { role: "system", content: "أنت مساعد عربي مفيد. أجب باللغة العربية الواضحة، وكن دقيقاً ومختصراً." },
          ...messages.slice(-20)
        ],
        max_tokens: Math.min(Number(body.max_tokens) || 512, 1024)
      });
      return Response.json({ reply: result.response || result.output_text || "لم أتمكن من إنشاء رد." }, { headers: corsHeaders });
    } catch (error) {
      return Response.json({ error: "AI engine unavailable", detail: error.message }, { status: 503, headers: corsHeaders });
    }
  }
};