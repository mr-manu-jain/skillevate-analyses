# main.py — full updated version

import json
import time
from app.resume_parser.extractor import extract_text
from app.resume_parser.sections import split_sections
from app.resume_parser.details import extract_details
from app.resume_parser.skills import extract_skills
from app.resume_parser.fallback import gemini_full_fallback, gemini_enrich_skills


class ResumeParser:
    def parse(self, pdf_path: str, inference: str = "ollama", enrich: bool = False) -> dict:
        """
        inference: "ollama" | "groq" | "none"
        enrich:    True = also run Gemini Flash skill enrichment after primary inference
        """
        start = time.time()

        extracted = extract_text(pdf_path)
        quality   = extracted["quality"]

        if quality == "bad":
            result = gemini_full_fallback(pdf_path)
            result["meta"] = {
                "extraction_method": "gemini_full_fallback",
                "quality":           quality,
                "sections_found":    list(result.get("raw_sections", {}).keys()),
                "parse_time_sec":    round(time.time() - start, 2),
            }
            return result

        sections = split_sections(extracted["text"])
        details  = extract_details(
            header_text  = sections.get("header", ""),
            summary_text = sections.get("summary", "")
        )
        skills = extract_skills(sections, inference=inference)

        if enrich or quality == "partial":
            work_text    = sections.get("experience", "") + "\n" + sections.get("projects", "")
            all_found    = skills["strong"] + skills["listed"]
            extra_skills = gemini_enrich_skills(work_text, all_found)
            merged = list(dict.fromkeys(
                s.strip().lower() for s in skills["strong"] + extra_skills
            ))
            skills["strong"] = sorted(merged)

        elapsed = round(time.time() - start, 2)

        return {
            "meta": {
                "extraction_method": extracted["method"],
                "quality":           quality,
                "sections_found":    list(sections.keys()),
                "parse_time_sec":    elapsed,
                "inference_engine":  inference,
            },
            "basic_details": details,
            "skills":        skills,
            "raw_sections": {
                k: v for k, v in sections.items()
                if k not in ("header",)
            }
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
    else:
        pdf_path = sys.argv[1] if len(sys.argv) > 1 else "tests/sample_resumes/resume.pdf"
        enrich   = "--enrich" in sys.argv
        result   = ResumeParser().parse(pdf_path, enrich=enrich)
        print(json.dumps(result, indent=2))