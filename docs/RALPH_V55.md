# V5.5 — verified native experts, fixed and adaptive ensembles

V5.4 passed all seven CPU tests on the user machine. V5.5 implements an
offline/local ensemble selection workflow without submitting GPU work.
The previous reported learned PESQ record remains 2.4800.

## Candidate models and inference
- Automatically shortlist 2–5 completed, verified V4-protocol native
  checkpoints from Ralph history, favoring distinct families, deduplicating
  checkpoint hashes. Default three; explicit --run folders also accepted.
  Existing challenge development scores shortlist experts. Old V2 records,
  smoke jobs, transductive runs and public wrappers are not admitted here.
- Fixed search: every nonempty uniform subset plus 25/75 and 75/25 pair
  blends (including individual-model fallback). For three experts this is
  13 fixed candidates.
- Clip gate: a small MLP over eight original noisy-audio features (log RMS,
  zero-crossing rate, crest, spectral flatness, four band energy fractions).
  Softmax weights sum to one across experts. Frequency gate predicts four
  independent simplexes for bands cut at 500, 1500 and 3500 Hz. Combines
  complex expert spectra and performs iSTFT. No clean targets, SNR labels,
  noise labels or speaker IDs are available to inference.
- Frozen experts produce cached waveforms in RAM; fit the two gates using
  normalized supervised L1 on 64 train clips. Real PCM16 PESQ selects
  among fixed and learned candidates on 32 other train clips with disjoint
  known speaker groups. No gate candidate sees validation targets during fit
  or mixture selection. The learned gate objective is L1, not direct PESQ.
- If the best mixture fails to beat the best single expert by >0.02 PESQ
  on the selection subset, keep the single expert. The margin is a heuristic,
  not a generalization guarantee. This is a global recipe fallback, not
  a clean-driven per-clip oracle.
- Base experts may have trained on these train clips; the gate split is
  speaker-disjoint only between its own fitting/selection sets. This is not
  true out-of-fold stacking or an independent generalization estimate.
  The final selected recipe may underperform on challenge validation.
- Save source/checkpoint hashes, exact train manifest hash, split indices,
  candidate scores and selected recipe. Strictly reload the selected bundle
  before reporting success. No raw audio is persisted.

## Local tests (no GPU)
```powershell
git pull --ff-only
py -m compileall -q src tools tests
py -m unittest discover -s tests -p "test_v55_ensemble.py" -v
```
CPU fixture tests use synthetic signals and mock metrics, not measured
speech enhancement PESQ. Real local expert runs and PESQ are required for
the following research command. Do not run it until at least two eligible
native experts exist and the final experiment schedule is reviewed.

```powershell
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
py -u tools/v55_ensemble.py --root "$DataRoot" --out runs/v55_ensemble_trial --max-experts 3 --gate-epochs 12 --validation
```

This runs locally on CPU by default. The optional --validation flag
evaluates only the frozen selected bundle on all 164 challenge clips;
it never reselects weights using that result. Use --run followed by
2–5 explicit completed local run directories if automatic discovery
cannot find enough native checkpoints. The directory must be new and
outside the dataset and expert folders. A failed run can leave a partial
candidate directory; only use a selected bundle with a successful
ensemble_report.json and validated artifacts.

The selected bundle is compatible with tools/submit_v5.py and its
PCM16 validation/inference flow. It requires the repository tools
(including v55_gating.py), PyTorch and existing dependencies. This is
not yet a standalone portable distribution; V5.7 will handle packaging.
CMGAN/MetricGAN/SepFormer and their cascades are deliberately not
admitted to the strict offline ensemble loader until public assets are
packaged and verified. No claim of improved PESQ until real runs.
V5.6 will integrate this local ensemble step into the continuous Ralph
schedule; V5.5 automatic expert discovery is implemented, not automatic
launch after every GPU experiment. Existing main remains untouched.
