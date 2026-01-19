#!/usr/bin/env python3
"""
VRP-RPD Genetic Programming Analyzer

Contains the GeneticAnalyzer class for:
- Building block extraction
- FFT frequency analysis
- Similarity clustering
- Candidate generation
"""

import numpy as np
from typing import Dict, List

# Optional CuPy import
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


class GeneticAnalyzer:
    """Genetic analysis for building blocks, FFT patterns, and similarity"""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and HAS_CUPY

    def analyze(self, solutions: List[Dict], cycle: int) -> Dict:
        """Run full analysis"""
        print(f"\n[GP] ========== GENETIC PROGRAMMING ANALYSIS STARTED ==========", flush=True)
        print(f"[GP] Cycle: {cycle}", flush=True)
        print(f"[GP] Input: {len(solutions)} solutions received for analysis", flush=True)

        if len(solutions) < 10:
            print(f"[GP] WARNING: Not enough solutions ({len(solutions)} < 10). Skipping analysis.", flush=True)
            print(f"[GP] ========== GP ANALYSIS SKIPPED ==========\n", flush=True)
            return {'building_blocks': [], 'fft_patterns': [], 'similarity_patterns': [], 'top_solutions': []}

        solutions = sorted(solutions, key=lambda x: x['fitness'])
        best_fitness = solutions[0]['fitness'] if solutions else float('inf')
        worst_fitness = solutions[-1]['fitness'] if solutions else float('inf')
        print(f"[GP] Fitness range: best={best_fitness:.2f}, worst={worst_fitness:.2f}", flush=True)

        print(f"[GP] Starting 3-phase analysis...", flush=True)

        print(f"[GP] Phase 1/3: Extracting building blocks...", flush=True)
        building_blocks = self._extract_building_blocks(solutions)
        print(f"[GP] Phase 1/3 complete: Found {len(building_blocks)} building blocks", flush=True)

        print(f"[GP] Phase 2/3: Running FFT frequency analysis...", flush=True)
        fft_patterns = self._analyze_fft(solutions, cycle)
        print(f"[GP] Phase 2/3 complete: Found {len(fft_patterns)} FFT patterns", flush=True)

        print(f"[GP] Phase 3/3: Computing similarity clusters...", flush=True)
        similarity_patterns = self._analyze_similarity(solutions, cycle)
        print(f"[GP] Phase 3/3 complete: Found {len(similarity_patterns)} similarity patterns", flush=True)

        print(f"[GP] ========== GP ANALYSIS COMPLETE ==========\n", flush=True)

        return {
            'building_blocks': building_blocks,
            'fft_patterns': fft_patterns,
            'similarity_patterns': similarity_patterns,
            'top_solutions': solutions[:100],
            'cycle': cycle
        }

    def _extract_building_blocks(self, solutions: List[Dict], block_sizes: List[int] = [3, 4, 5, 6]) -> List[Dict]:
        """Extract common patterns from top solutions"""
        blocks = []
        top_n = min(50, len(solutions))
        top_sols = solutions[:top_n]

        chroms = []
        for sol in top_sols:
            c = sol['chromosome']
            if len(c.shape) == 1 and len(c) % 4 == 0:
                c = c.reshape(-1, 4)
            chroms.append(c)

        if not chroms:
            return blocks

        min_len = min(len(c) for c in chroms)

        for block_size in block_sizes:
            if min_len < block_size:
                continue

            for start in range(0, min_len - block_size + 1, max(1, block_size // 2)):
                end = start + block_size

                block_values = []
                for c in chroms:
                    if len(c.shape) == 2:
                        block_values.append(c[start:end].flatten())
                    else:
                        block_values.append(c[start:end])

                block_array = np.array(block_values, dtype=np.float32)
                mean_block = np.mean(block_array, axis=0)
                variance = float(np.mean(np.var(block_array, axis=0)))

                distances = np.sqrt(np.sum((block_array - mean_block) ** 2, axis=1))
                threshold = np.median(distances)
                frequency = float(np.mean(distances < threshold))

                strength = frequency / (variance + 0.01)

                if strength > 1.0:
                    blocks.append({
                        'start_pos': start,
                        'end_pos': end,
                        'block_size': block_size,
                        'pattern': mean_block.tolist(),
                        'variance': variance,
                        'frequency': frequency,
                        'strength': strength
                    })

        blocks.sort(key=lambda x: x['strength'], reverse=True)
        return blocks[:30]

    def _analyze_fft(self, solutions: List[Dict], cycle: int) -> List[Dict]:
        """FFT analysis on chromosome structure"""
        patterns = []

        for idx, sol in enumerate(solutions[:50]):
            chrom = sol['chromosome']
            if len(chrom.shape) > 1:
                chrom = chrom.flatten()

            if len(chrom) < 16:
                continue

            data = chrom[:64]
            n = 64
            padded = np.zeros(n, dtype=np.float32)
            padded[:len(data)] = data
            fft_result = np.fft.fft(padded)
            power = np.abs(fft_result) ** 2

            strength = float(np.std(power))
            dominant_freq = int(np.argmax(power[1:n//2])) + 1
            peak_power = float(np.max(power[1:n//2]))

            if strength > 0.01:
                patterns.append({
                    'pattern_id': f'fft_sol{sol["id"]}_cycle{cycle}',
                    'sol_id': sol['id'],
                    'fitness': sol['fitness'],
                    'strength': strength,
                    'dominant_frequency': dominant_freq,
                    'peak_power': peak_power
                })

        patterns.sort(key=lambda x: x['strength'], reverse=True)
        return patterns[:20]

    def _analyze_similarity(self, solutions: List[Dict], cycle: int) -> List[Dict]:
        """Compute similarity statistics"""
        patterns = []
        top_n = min(100, len(solutions))
        top_sols = solutions[:top_n]

        chroms = []
        for sol in top_sols:
            c = sol['chromosome']
            if len(c.shape) > 1:
                c = c.flatten()
            chroms.append(c[:200])

        max_len = max(len(c) for c in chroms)
        chroms = [np.pad(c, (0, max_len - len(c))) for c in chroms]
        chrom_matrix = np.array(chroms, dtype=np.float32)

        n = len(chroms)
        all_distances = []

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((chrom_matrix[i] - chrom_matrix[j]) ** 2))
                all_distances.append(dist)

        if all_distances:
            all_distances = np.array(all_distances)
            threshold = np.percentile(all_distances, 25)

            patterns.append({
                'pattern_id': f'similarity_cycle{cycle}',
                'type': 'similarity_cluster',
                'threshold': float(threshold),
                'avg_distance': float(np.mean(all_distances)),
                'min_distance': float(np.min(all_distances)),
                'n_close_pairs': int(np.sum(all_distances < threshold))
            })

        return patterns

    def generate_candidates(self, analysis: Dict, n_candidates: int = 20, m: int = 6) -> List[np.ndarray]:
        """Generate new candidates based on analysis"""
        print(f"\n[GP] ========== CANDIDATE GENERATION STARTED ==========", flush=True)
        print(f"[GP] Generating {n_candidates} new candidate solutions", flush=True)

        candidates = []

        top_solutions = analysis.get('top_solutions', [])
        building_blocks = analysis.get('building_blocks', [])

        print(f"[GP] Available data: {len(top_solutions)} top solutions, {len(building_blocks)} building blocks", flush=True)

        if not top_solutions:
            print(f"[GP] WARNING: No top solutions available. Cannot generate candidates.", flush=True)
            print(f"[GP] ========== CANDIDATE GENERATION FAILED ==========\n", flush=True)
            return candidates

        n_block = n_candidates // 3
        n_interpolate = n_candidates // 3
        n_mutate = n_candidates - n_block - n_interpolate

        print(f"[GP] Generation strategy: {n_block} block-based, {n_interpolate} interpolated, {n_mutate} mutated", flush=True)

        # Strategy 1: Building block combination
        print(f"[GP] Strategy 1: Creating {n_block} candidates via building block combination...", flush=True)
        for _ in range(n_block):
            base_idx = np.random.randint(0, min(20, len(top_solutions)))
            chrom = top_solutions[base_idx]['chromosome'].copy()

            if building_blocks and len(chrom.shape) == 2:
                n_insert = np.random.randint(1, min(4, len(building_blocks)) + 1)
                selected_blocks = np.random.choice(len(building_blocks), n_insert, replace=False)

                for b_idx in selected_blocks:
                    block = building_blocks[b_idx]
                    start, end = block['start_pos'], block['end_pos']
                    pattern = np.array(block['pattern'])

                    if end <= len(chrom):
                        block_shape = chrom[start:end].shape
                        if pattern.size >= np.prod(block_shape):
                            mix = np.random.uniform(0.3, 0.8)
                            pattern_reshaped = pattern[:np.prod(block_shape)].reshape(block_shape)
                            chrom[start:end] = (1 - mix) * chrom[start:end] + mix * pattern_reshaped

            self._fix_constraints(chrom, m)
            candidates.append(chrom.astype(np.float32))
        print(f"[GP] Strategy 1 complete: {n_block} block-based candidates created", flush=True)

        # Strategy 2: Interpolation
        print(f"[GP] Strategy 2: Creating {n_interpolate} candidates via solution interpolation...", flush=True)
        for _ in range(n_interpolate):
            if len(top_solutions) >= 2:
                idx1, idx2 = np.random.choice(min(30, len(top_solutions)), 2, replace=False)
                c1 = top_solutions[idx1]['chromosome']
                c2 = top_solutions[idx2]['chromosome']

                alpha = np.random.uniform(-0.2, 1.2)
                chrom = alpha * c1 + (1 - alpha) * c2

                self._fix_constraints(chrom, m)
                candidates.append(chrom.astype(np.float32))
        print(f"[GP] Strategy 2 complete: {n_interpolate} interpolated candidates created", flush=True)

        # Strategy 3: Guided mutation
        print(f"[GP] Strategy 3: Creating {n_mutate} candidates via guided mutation...", flush=True)
        for _ in range(n_mutate):
            base = top_solutions[np.random.randint(0, min(10, len(top_solutions)))]['chromosome'].copy()
            noise = np.random.randn(*base.shape).astype(np.float32) * 0.05
            chrom = base + noise

            self._fix_constraints(chrom, m)
            candidates.append(chrom.astype(np.float32))
        print(f"[GP] Strategy 3 complete: {n_mutate} mutated candidates created", flush=True)

        print(f"[GP] Total candidates generated: {len(candidates)}", flush=True)
        print(f"[GP] ========== CANDIDATE GENERATION COMPLETE ==========\n", flush=True)

        return candidates

    def _fix_constraints(self, chrom: np.ndarray, m: int):
        """Fix VRP constraints in-place"""
        if len(chrom.shape) == 2 and chrom.shape[1] == 4:
            chrom[:, 0] = np.clip(np.round(chrom[:, 0]), 0, m - 1)
            chrom[:, 1] = np.clip(np.round(chrom[:, 1]), 0, m - 1)
            chrom[:, 3] = np.maximum(chrom[:, 3], chrom[:, 2] + 0.01)
