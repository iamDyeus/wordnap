"""Candidate ranking and variation generation with filter-then-rank strategy."""

from itertools import product

from sentence_mixer.models.schemas import RankingConfig, WordCandidate


class Ranker:
    """Two-pass filter-then-rank candidate selection.

    Pass 1: Filter out candidates below quality thresholds.
    Pass 2: Score remaining candidates with composite weighted scoring.
    """

    def __init__(self, config: RankingConfig):
        self._config = config

    def filter_candidates(
        self, candidates: list[WordCandidate]
    ) -> list[WordCandidate]:
        """Pass 1: Remove candidates below quality thresholds.

        Filters on:
        - confidence >= min_confidence (0.3)
        - min_duration <= duration <= max_duration (0.03s to 3.0s)

        If filtering removes ALL candidates, returns the original list
        as fallback to avoid total failure.

        Args:
            candidates: List of candidates for a single token.

        Returns:
            Filtered list, or original list if filtering was too aggressive.
        """
        filtered = [
            c
            for c in candidates
            if c.word.confidence >= self._config.min_confidence
            and self._config.min_duration <= c.duration <= self._config.max_duration
        ]
        return filtered if filtered else candidates

    def compute_duration_score(self, duration: float) -> float:
        """Compute duration reasonableness score using piecewise linear function.

        Returns 1.0 for durations in [ideal_min, ideal_max].
        Linearly decreases to 0.0 at filter boundaries.

        - duration in [0.1, 1.5] -> 1.0
        - duration in [0.03, 0.1) -> linear from 0.0 to 1.0
        - duration in (1.5, 3.0] -> linear from 1.0 to 0.0
        - duration < 0.03 or > 3.0 -> 0.0

        Args:
            duration: Clip duration in seconds.

        Returns:
            Score in [0.0, 1.0].
        """
        config = self._config
        if config.ideal_duration_min <= duration <= config.ideal_duration_max:
            return 1.0
        elif duration < config.ideal_duration_min:
            if duration <= config.min_duration:
                return 0.0
            return (duration - config.min_duration) / (
                config.ideal_duration_min - config.min_duration
            )
        else:
            if duration >= config.max_duration:
                return 0.0
            return (config.max_duration - duration) / (
                config.max_duration - config.ideal_duration_max
            )

    def compute_boundary_quality(self, candidate: WordCandidate) -> float:
        """Compute boundary quality score.

        Measures proximity to natural silence boundaries by approximating
        the gap between the word start_time and a padded position.
        Larger gap suggests the word starts/ends at a natural silence boundary.

        Approximation: boundary_quality = (start_gap + end_gap) / (2 * max_padding)

        Args:
            candidate: WordCandidate with timing information.

        Returns:
            Score in [0.0, 1.0].
        """
        max_padding = 0.15  # reasonable max reference
        start_gap = min(max_padding, candidate.word.start_time)
        end_gap = min(
            max_padding, candidate.video.duration - candidate.word.end_time
        )

        quality = (start_gap + end_gap) / (2 * max_padding)
        return max(0.0, min(1.0, quality))

    def compute_diversity_score(
        self, candidate: WordCandidate, used_sources: set[int | None]
    ) -> float:
        """Compute source diversity score.

        Returns 1.0 if candidate is from an unused source, 0.0 otherwise.

        Args:
            candidate: The candidate to score.
            used_sources: Set of video IDs already selected in this variation.

        Returns:
            1.0 or 0.0.
        """
        if candidate.video.id not in used_sources:
            return 1.0
        return 0.0

    def score_candidate(
        self,
        candidate: WordCandidate,
        used_sources: set[int | None] | None = None,
    ) -> float:
        """Pass 2: Compute composite weighted score.

        Score = confidence * 0.35 + duration_score * 0.25
              + boundary_quality * 0.20 + diversity * 0.20

        Args:
            candidate: The candidate to score.
            used_sources: Sources already used in this variation (default empty).

        Returns:
            Composite score in [0.0, 1.0].
        """
        if used_sources is None:
            used_sources = set()

        config = self._config
        confidence_score = candidate.word.confidence
        duration_score = self.compute_duration_score(candidate.duration)
        boundary_score = self.compute_boundary_quality(candidate)
        diversity_score = self.compute_diversity_score(candidate, used_sources)

        score = (
            confidence_score * config.confidence_weight
            + duration_score * config.duration_weight
            + boundary_score * config.boundary_quality_weight
            + diversity_score * config.diversity_weight
        )
        return max(0.0, min(1.0, score))

    def rank(
        self,
        candidates_per_token: dict[str, list[WordCandidate]],
        num_variations: int,
    ) -> list[list[WordCandidate]]:
        """Produce N ranked variations using filter-then-rank strategy.

        Each variation is a distinct combination of word candidates. Variations
        are ordered by total score in descending order.

        Args:
            candidates_per_token: Dict mapping token to its candidate list.
            num_variations: Desired number of variations to produce.

        Returns:
            List of variations, each being a list of selected WordCandidates.
        """
        if not candidates_per_token:
            return []

        tokens = list(candidates_per_token.keys())

        # Pass 1: Filter candidates per token
        filtered_per_token: dict[str, list[WordCandidate]] = {}
        for token in tokens:
            filtered_per_token[token] = self.filter_candidates(
                candidates_per_token[token]
            )

        # Score all candidates (with empty used_sources for initial scoring)
        for token in tokens:
            for candidate in filtered_per_token[token]:
                candidate.score = self.score_candidate(candidate)

        # Sort candidates per token by score descending
        for token in tokens:
            filtered_per_token[token].sort(
                key=lambda c: c.score, reverse=True
            )

        # Calculate max possible unique combinations
        max_combinations = 1
        for token in tokens:
            count = len(filtered_per_token[token])
            max_combinations *= count
            if max_combinations > num_variations * 100:
                break

        target_count = min(num_variations, max_combinations)

        # Generate variations
        candidate_counts = [len(filtered_per_token[t]) for t in tokens]
        total_combos = 1
        for c in candidate_counts:
            total_combos *= c
            if total_combos > 10000:
                break

        if total_combos <= 10000:
            variations = self._exhaustive_rank(
                filtered_per_token, tokens, target_count
            )
        else:
            variations = self._greedy_rank(
                filtered_per_token, tokens, target_count, set()
            )

        # Sort variations by total score descending
        variations.sort(
            key=lambda v: sum(c.score for c in v), reverse=True
        )

        return variations

    def _exhaustive_rank(
        self,
        candidates_per_token: dict[str, list[WordCandidate]],
        tokens: list[str],
        target_count: int,
    ) -> list[list[WordCandidate]]:
        """Generate all combinations, score with diversity, return top N."""
        candidate_lists = [candidates_per_token[t] for t in tokens]

        scored_variations: list[tuple[float, list[WordCandidate]]] = []

        for combo in product(*candidate_lists):
            selection = list(combo)
            total_score = self._compute_variation_score(selection)
            scored_variations.append((total_score, selection))

        # Sort by total score descending
        scored_variations.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate and pick top N
        variations: list[list[WordCandidate]] = []
        used_combinations: set[tuple[int | None, ...]] = set()

        for _, selection in scored_variations:
            combo_key = tuple(w.word.id for w in selection)
            if combo_key not in used_combinations:
                used_combinations.add(combo_key)
                variations.append(selection)
                if len(variations) >= target_count:
                    break

        return variations

    def _greedy_rank(
        self,
        candidates_per_token: dict[str, list[WordCandidate]],
        tokens: list[str],
        target_count: int,
        used_combinations: set[tuple[int | None, ...]],
    ) -> list[list[WordCandidate]]:
        """Greedy variation generation for large candidate spaces."""
        variations: list[list[WordCandidate]] = []

        max_attempts = target_count * 10

        for attempt in range(max_attempts):
            if len(variations) >= target_count:
                break

            selection: list[WordCandidate] = []
            used_sources: set[int | None] = set()

            for token in tokens:
                candidates = candidates_per_token[token]
                best = self._select_best_candidate(
                    candidates,
                    used_sources=used_sources,
                    previous_speaker=(
                        selection[-1].word.speaker if selection else None
                    ),
                    offset=attempt,
                )
                selection.append(best)
                used_sources.add(best.video.id)

            combo_key = tuple(w.word.id for w in selection)
            if combo_key not in used_combinations:
                used_combinations.add(combo_key)
                variations.append(selection)

        return variations

    def _select_best_candidate(
        self,
        candidates: list[WordCandidate],
        used_sources: set[int | None],
        previous_speaker: str | None,
        offset: int = 0,
    ) -> WordCandidate:
        """Select the best candidate considering diversity and speaker preference."""
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        if len(candidates) == 1:
            return candidates[0]

        effective_offset = offset % len(candidates)

        # Score candidates with diversity context
        scored = []
        for i, candidate in enumerate(candidates):
            score = self.score_candidate(candidate, used_sources)

            # Prefer same speaker if configured
            if (
                self._config.prefer_same_speaker
                and previous_speaker is not None
                and candidate.word.speaker == previous_speaker
            ):
                score += 0.05

            scored.append((score, i, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)

        pick_index = effective_offset % len(scored)
        return scored[pick_index][2]

    def rank_round_robin(
        self,
        candidates_per_token: dict[str, list[WordCandidate]],
        num_variations: int,
    ) -> list[list[WordCandidate]]:
        """Produce variations with round-robin source rotation.

        For each word position, selects the best candidate from the next
        source in the rotation. Cycles through all available video sources.

        Args:
            candidates_per_token: Dict mapping token to its candidate list.
            num_variations: Number of variations to produce.

        Returns:
            List of variations with enforced source cycling.
        """
        if not candidates_per_token:
            return []

        tokens = list(candidates_per_token.keys())

        # Filter candidates
        filtered_per_token: dict[str, list[WordCandidate]] = {}
        for token in tokens:
            filtered_per_token[token] = self.filter_candidates(
                candidates_per_token[token]
            )

        # Discover all unique video sources across all candidates
        all_sources: list[int] = []
        seen_sources: set[int] = set()
        for token in tokens:
            for candidate in filtered_per_token[token]:
                vid_id = candidate.video.id
                if vid_id is not None and vid_id not in seen_sources:
                    seen_sources.add(vid_id)
                    all_sources.append(vid_id)

        if not all_sources:
            # Fallback to normal ranking
            return self.rank(candidates_per_token, num_variations)

        # Generate variations with different starting positions in the rotation
        variations: list[list[WordCandidate]] = []
        used_combinations: set[tuple[int | None, ...]] = set()

        for start_offset in range(min(num_variations, len(all_sources))):
            selection: list[WordCandidate] = []

            for i, token in enumerate(tokens):
                # Determine which source to prefer for this position
                target_source_idx = (i + start_offset) % len(all_sources)
                target_source = all_sources[target_source_idx]

                # Find best candidate from the target source
                best = self._pick_from_source(
                    filtered_per_token[token], target_source
                )
                if best is None:
                    # Fallback: pick highest scoring candidate from any source
                    best = self._pick_best_available(filtered_per_token[token])

                if best is not None:
                    best.score = self.score_candidate(best)
                    selection.append(best)

            if selection and len(selection) == len(tokens):
                combo_key = tuple(w.word.id for w in selection)
                if combo_key not in used_combinations:
                    used_combinations.add(combo_key)
                    variations.append(selection)

        # If we couldn't produce enough variations from rotation alone,
        # fill with additional offset patterns
        if len(variations) < num_variations:
            for offset in range(
                len(all_sources), num_variations * len(all_sources)
            ):
                if len(variations) >= num_variations:
                    break

                selection = []
                for i, token in enumerate(tokens):
                    target_source_idx = (i + offset) % len(all_sources)
                    target_source = all_sources[target_source_idx]
                    best = self._pick_from_source(
                        filtered_per_token[token], target_source
                    )
                    if best is None:
                        best = self._pick_best_available(
                            filtered_per_token[token]
                        )
                    if best is not None:
                        best.score = self.score_candidate(best)
                        selection.append(best)

                if selection and len(selection) == len(tokens):
                    combo_key = tuple(w.word.id for w in selection)
                    if combo_key not in used_combinations:
                        used_combinations.add(combo_key)
                        variations.append(selection)

        return variations

    def _pick_from_source(
        self, candidates: list[WordCandidate], target_source: int
    ) -> WordCandidate | None:
        """Pick the highest-scoring candidate from a specific source video.

        Returns None if no candidate from that source exists.
        """
        source_candidates = [
            c for c in candidates if c.video.id == target_source
        ]
        if not source_candidates:
            return None
        # Score and pick best
        for c in source_candidates:
            c.score = self.score_candidate(c)
        return max(source_candidates, key=lambda c: c.score)

    def _pick_best_available(
        self, candidates: list[WordCandidate]
    ) -> WordCandidate | None:
        """Pick the highest-scoring candidate from any source."""
        if not candidates:
            return None
        for c in candidates:
            c.score = self.score_candidate(c)
        return max(candidates, key=lambda c: c.score)

    def _compute_variation_score(
        self, selection: list[WordCandidate]
    ) -> float:
        """Compute total score for a variation including diversity."""
        total = 0.0
        used_sources: set[int | None] = set()
        previous_speaker: str | None = None

        for candidate in selection:
            score = self.score_candidate(candidate, used_sources)

            # Same speaker preference
            if (
                self._config.prefer_same_speaker
                and previous_speaker is not None
                and candidate.word.speaker == previous_speaker
            ):
                score += 0.05

            candidate.score = score
            total += score
            used_sources.add(candidate.video.id)
            previous_speaker = candidate.word.speaker

        return total
