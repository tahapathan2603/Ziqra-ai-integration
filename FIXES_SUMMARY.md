# Audio Analysis Contradiction Fixes – Summary

## Overview
Fixed contradictory and duplicated analysis across Fluency, Pronunciation, MTI, Intonation, and Engagement modules so the report speaks with one voice. All fixes verified end-to-end on real audio (`chunk_1 copy.wav`).

## Changes by Module

### 1. MTI – Eliminated Double-Counting (Real Scoring Bug)
- **Problem:** Errors like `/θ/→/t/` flagged in two panels (Phoneme Substitutions + Consonant Patterns) with different severities, penalized twice.
- **Fix:** 
  - Created `backend/feature_extractors/audio/phoneme_patterns.py` – shared severity tables for vowel/consonant patterns
  - Removed redundant `phoneme_substitution.py` analysis role (kept only `build_word_timestamps`)
  - Every error now classified & counted exactly once in Consonant Patterns panel
  - Lowered generic (uncurated) vowel substitution severity from "medium" → "low" (schwa shifts on function words aren't clarity issues)

### 2. Pronunciation – Honest Scoring & Severity Alignment
- **Problem:** Stress accuracy (placeholder, always 100%) inflated the composite score; same error read different severity in Pronunciation vs MTI.
- **Fix:**
  - Exclude stress_accuracy from composite: `score = (phoneme_accuracy + rhythm_score) / 2`
  - Align word severity: take max of count-based severity and known-pattern severity from shared `phoneme_patterns.py`
  - Rename rhythm issue `"monotone pacing"` → `"uniform word timing"` (monotone belongs to intonation only)
  - UI note: stress marked `[PLACEHOLDER detection — excluded from score]`

### 3. Intonation – One Flatness Verdict, No Self-Contradiction
- **Problem:** Pitch said "healthy", monotonicity said "flat"; three modules claimed the response "fades at the end".
- **Fix:**
  - Replace energy threshold (25th percentile, artificially flagged ~25% of every recording) with ratio-based: low = `rms < 0.5 × median(speech_rms)`
  - Restructure energy observations as mutually exclusive branch (consistent *or* fades *or* dips, never multiple)
  - Reconcile pitch vs monotonicity: if monotonicity low, rewrite pitch's "healthy" to "varies overall, but delivery stays flat for sustained stretches"
  - Deliver vocab: `"Expressive"` / `"Moderately expressive"` / `"Somewhat flat"` / `"Flat"` (distinct from engagement's "Engaging" scale)
  - Add `delivery_label` to report output; UI reads it instead of re-deriving

### 4. Engagement – Internally Consistent, Synthesis-Only Panels
- **Problem:** Conflicting strengths/improvements ("good pitch variation" + "sounds monotone"); score (93) vs timeline ("Low") contradiction.
- **Fix:**
  - Speaking dynamics: single vocal-variation verdict (one strength *or* one issue, never both); fold emphasis complaint into monotone issue when applicable
  - Score↔timeline coherence: compute problem section coverage, apply penalty to blended score, cap engagement level so it never exceeds what timeline supports
  - Dedupe improvements: merge "Pronunciation issues on key word(s)" + "Clarity risk on word(s)" into one deduped line
  - Energy/Pause panels: synthesis-only (no re-listing upstream segments); full detail in Raw JSON accordion
  - Result on your file: 93 (contradicting timeline) → 65 (coherent with "low" window)

### 5. UI Plumbing
- Removed MTI Phoneme Substitutions formatter, component, and output tuple slot
- Output tuple shrinks 38 → 37 slots
- Updated intonation summary to read `delivery_label` from report
- Updated pronunciation summary to note stress is excluded from score
- Updated MTI description to clarify each error classified once

## Testing
- ✅ Syntax: all 15 touched files parse cleanly
- ✅ Synthetic logic: 22 pure-function tests pass (MTI single-counting, severity alignment, dynamics gating, energy exclusivity, engagement coverage/cap, etc.)
- ✅ End-to-end: full pipeline on real audio passes 10 contradiction assertions:
  - MTI phoneme_substitutions removed
  - pron severity ≥ MTI severity for all shared words (including "to", now aligned at "low")
  - No conflicting vocal-variation strength/issue
  - Any low timeline window → level not "highly_engaging"
  - Majority low windows → level in {needs_improvement, low_engagement}
  - Improvements deduped (no split pron/clarity lines)
  - "europe" still correctly flagged high
  - No pitch-healthy contradiction with low monotonicity
  - Intonation label doesn't reuse "Engaging" vocabulary
  - Stress correctly excluded from pronunciation score

## Impact
Report is now **one-directional**: every claim made once, from one source of truth. Same word shows the same severity everywhere. No strength contradicts an improvement in the same panel. Engagement score, label, and timeline move together by construction.
