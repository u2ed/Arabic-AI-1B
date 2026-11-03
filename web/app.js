const form=document.getElementById("chat-form");
const input=document.getElementById("message");
const chat=document.getElementById("chat");
const send=document.getElementById("send");
const statusEl=document.getElementById("status");
let messages=[];
function bubble(role,text){const el=document.createElement("div");el.className="bubble "+role;el.textContent=text;chat.appendChild(el);chat.scrollTop=chat.scrollHeight;return el}
form.addEventListener("submit",async e=>{
 e.preventDefault();const text=input.value.trim();if(!text)return;
 const api=window.ARABIC_AI_CONFIG?.API_URL;
 if(!api){statusEl.textContent="بانتظار إعداد المحرك";bubble("assistant","الواجهة جاهزة. لم نربط محرك الذكاء الاصطناعي بعد.");return}
 document.getElementById("welcome")?.remove();
 bubble("user",text);messages.push({role:"user",content:text});input.value="";send.disabled=true;statusEl.textContent="يفكر...";
 const waiting=bubble("assistant","جاري التفكير...");
 try{const res=await fetch(api,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({messages,max_tokens:512})});
 const data=await res.json();if(!res.ok)throw new Error(data.error||"تعذر الاتصال بالمحرك");
 const reply=data.reply||data.response||"لم يصل رد.";waiting.textContent=reply;messages.push({role:"assistant",content:reply});statusEl.textContent="متصل";
 }catch(err){waiting.textContent="المحرك غير متاح حالياً: "+err.message;statusEl.textContent="غير متصل"}
 finally{send.disabled=false;input.focus()}
});