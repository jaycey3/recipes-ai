import streamlit as st
import tempfile
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from src.pipeline import process_video, save_recipe

st.set_page_config(page_title="Recipe Extractor", page_icon="🍎", layout="wide")

st.title("🍎 Recipe Extractor")
st.markdown("Upload a cooking video, and I'll generate a structured recipe for you.")

with st.sidebar:
    st.header("⚙️ Settings")
    frame_fps = st.slider("Frames per second", min_value=0.5, max_value=2.0, value=1.0, step=0.5, help="Higher = more frames analyzed, but slower processing.")
    detection_threshold = st.slider("Detection threshold", min_value=0.3, max_value=0.9, value=0.5, step=0.05, help="Confidence threshold for ingredient detection.")
    whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=2, help="Larger = better transcription, but slower processing.")
    language = st.selectbox("Language of the video", options=[("nl", "Nederlands"), ("en", "English"), ("auto", "Detect automatically")], index=0, format_func=lambda x: x[1], help="Language for transcription. 'Detect automatically' uses Whisper's language detection.")
    language_code = language[0]

uploaded = st.file_uploader("Upload a cooking video (mp4, mov, avi)", type=["mp4", "mov", "avi", "mkv"])

if uploaded is not None:
    video_col, _ = st.columns([1, 4])
    with video_col:
        st.video(uploaded)

    if st.button("🚀 Start Processing", type="primary"):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        progress_text = st.empty()
        progress_bar = st.progress(0)

        step_count = [0]
        total_steps = 7

        def progress_callback(msg: str):
            progress_text.text(msg)
            step_count[0] += 1
            progress_bar.progress(min(step_count[0] / total_steps, 1.0))

        try:
            recipe = process_video(tmp_path, frame_fps=frame_fps, detection_threshold=detection_threshold, whisper_size=whisper_size, language=language_code, progress_callback=progress_callback)
            progress_bar.progress(1.0)
            progress_text.text("✅ Done!")

            output_name = os.path.splitext(uploaded.name)[0] + ".json"
            output_path = os.path.join("data/output_recipes", output_name)
            save_recipe(recipe, output_path)

            if "_error" in recipe:
                st.error(f"An error occurred: {recipe['_error']}")
                st.code(recipe.get("_raw_response", "No raw response available"), language="json")
            else:
                st.success(f"Recipe saved to `{output_path}`")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.header(recipe.get("title", "Untitled Recipe"))

                    if recipe.get("servings"):
                        st.caption(f"🍽️ {recipe['servings']}")
                    if recipe.get("total_time"):
                        st.caption(f"⏱️ {recipe['total_time']} minutes")

                    st.subheader("📝 Steps")
                    for i, step in enumerate(recipe.get("steps", []), 1):
                        st.markdown(f"**{i}**. {step}")

                with col2:
                    st.subheader("🛒 Ingredients")
                    for ingredient in recipe.get("ingredients", []):
                        st.markdown(f"- **{ingredient.get("amount", "")}** {ingredient.get("name", "")}")
                
                if recipe.get("notes"):
                    for note in recipe["notes"]:
                        st.info(f"💡 **Tip:** {note}")

                with st.expander("Debug info"):
                    debug = recipe.get("_debug", {})
                    st.write(f"Frames analyzed: {debug.get('num_frames', 'N/A')}")
                    st.write(f"Transcript segments: {debug.get('num_transcript_segments', 'N/A')}")
                    st.json(debug.get("merged_input", {}))

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.exception(e)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)