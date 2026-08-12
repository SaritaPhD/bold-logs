# BOLD Colab package

Run order (notebook does all of this): open BOLD_colab.ipynb in Google Colab,
set Runtime -> T4 GPU, run cells top to bottom, upload this zip when asked.

Priority: embed7 on bgl_chrono (the paper-deciding experiment, ~5-10 min),
then LogBERT on bgl_chrono (~15-30 min on T4), then the remaining three
LogBERT configs as session time allows. Free-tier sessions can disconnect;
every run is independent and re-runnable, so nothing is lost.

When done: download colab_results.zip (last cell) and place it inside the
connected "Log Anomaly Detection Paper" folder. I will verify and merge.

Rules: send tracebacks instead of patching scripts; never edit result JSONs.
