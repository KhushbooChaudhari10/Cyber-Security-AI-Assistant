import os
import json
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN not set in environment variables")
    print("Please create a .env file with your HuggingFace token:")
    print("HF_TOKEN=your_huggingface_token_here")
    raise ValueError("HF_TOKEN not set in environment variables")

client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)


class MemoryAgentHF:
    def __init__(self, model="openai/gpt-oss-20b:nebius",
                 retention=timedelta(days=1),
                 summary_interval: int = 5,
                 recent_keep: int = 4,
                 memory_folder: str = "memory"):
        self.model = model
        self.retention = retention
        self.summary_interval = summary_interval
        self.recent_keep = recent_keep
        self.memory_folder = memory_folder

        os.makedirs(self.memory_folder, exist_ok=True)

        self.chat_history = []  # stores all messages {"role": ..., "content": ...}
        self.summary = ""       # running summary

    # ---------------- Utility ----------------
    def _save_summary(self):
        """Save running summary to persistent JSON file."""
        try:
            summary_file = os.path.join(self.memory_folder, "running_summary.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump({"summary": self.summary}, f, indent=2, ensure_ascii=False)
            print(f"✅ Running summary saved to {summary_file}")
        except Exception as e:
            print(f"⚠️ Failed to save summary: {e}")

    def _summarize_history(self, min_messages: int = None):
        if min_messages is None:
            min_messages = self.summary_interval

        total = len(self.chat_history)
        if total < min_messages:
            return

        cutoff = max(0, total - self.recent_keep)
        old_msgs = self.chat_history[:cutoff]
        history_blob = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in old_msgs]).strip()
        if not history_blob:
            return

        prompt = (
            "Summarize the important facts, decisions, and user preferences from the following conversation. "
            "Produce a short paragraph (2-4 sentences) that can be used as long-term memory.\n\n"
            f"{history_blob}\n\n"
            "Return only the summary (no extra text)."
        )

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=400
            )
            summary_text = resp.choices[0].message.content.strip()
        except Exception as e:
            print("⚠️ Summarization failed:", e)
            summary_text = ""

        if summary_text:
            self.summary = self._merge_summaries(self.summary, summary_text) if self.summary else summary_text
            self.chat_history = self.chat_history[cutoff:]
            self._save_summary()  # <-- Save summary immediately
            print("📝 Periodic summary updated.")

    def _merge_summaries(self, old_summary: str, new_summary: str) -> str:
        return old_summary + "\n\n[Update] " + new_summary

    # ---------------- Main response ----------------
    def get_response(self, user_message: str) -> str:
        if not user_message.strip():
            return "Please enter a message."

        # Add user message
        self.chat_history.append({"role": "user", "content": user_message})

        # Periodic summarization
        self._summarize_history()

        # Build messages for HF API
        messages = []
        if self.summary:
            messages.append({"role": "system", "content": f"Summary of previous conversation: {self.summary}"})
        messages.extend(self.chat_history)

        # Load system instructions from file
        try:
            prompt_file_path = os.path.join(os.path.dirname(__file__), "..", "static", "prompt.md")
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                system_instructions = f.read()
        except FileNotFoundError:
            # Fallback prompt if file not found
            system_instructions = "You are Cyra, a cybersecurity education assistant focused on defensive security practices."
        messages.insert(0, {"role": "system", "content": system_instructions})

        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=700
            )
            ai_response = completion.choices[0].message.content.strip()
            self.chat_history.append({"role": "assistant", "content": ai_response})
            return ai_response
        except Exception as e:
            print("⚠️ Hugging Face request failed:", e)
            if "401" in str(e) or "authorization" in str(e).lower():
                return "Authentication error: Please check your HuggingFace token configuration."
            elif "timeout" in str(e).lower():
                return "Request timeout: The AI service is taking too long to respond. Please try again."
            else:
                return "Sorry, I'm having trouble connecting to the AI service. Please try again later."

    # ---------------- Export memory ----------------
    def export_memory(self):
        self._summarize_history(min_messages=2)
        filename = f"chat_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        file_path = os.path.join(self.memory_folder, filename)
        data = {
            "summary": self.summary,
            "messages": self.chat_history
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"✅ Chat memory exported to {file_path}")
        return file_path
