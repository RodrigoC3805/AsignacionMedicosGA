from __future__ import annotations

import json
import random
import statistics
import threading
import urllib.parse
import os
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Tuple


# ============================================================
# MODELOS DE DATOS
# ============================================================

@dataclass
class Doctor:
    id: int
    name: str
    preferences: List[int]   # IDs de establecimientos (1-8), en orden de preferencia

@dataclass
class Establishment:
    id: int
    name: str
    risk_level: str
    capacity_max: int
    min_quota: int
    bonus_cost: int

@dataclass
class FitnessBreakdown:
    fitness: float
    satisfaction_total: int
    penalty_capacity: int
    penalty_epidemiological: int
    penalty_budget: int
    total_cost: int
    budget_excess: int
    assigned_per_establishment: Dict[int, int]


# ============================================================
# INSTANCIA FIJA DEL DOCUMENTO
# ============================================================

DOCUMENT_ESTABLISHMENTS = [
    Establishment(1, "Posta Chumbivilcas", "Alto",  5, 4, 2500),
    Establishment(2, "C.S. Coré",          "Alto",  4, 3, 2500),
    Establishment(3, "Posta Putina",        "Alto",  5, 4, 2200),
    Establishment(4, "C.S. Acomayo",        "Medio", 4, 2, 1500),
    Establishment(5, "C.S. Paruro",         "Medio", 5, 2, 1500),
    Establishment(6, "C.S. Quiquijana",     "Medio", 4, 2, 1300),
    Establishment(7, "C.S. Oropesa",        "Bajo",  6, 1, 1000),
    Establishment(8, "C.S. Lucre",          "Bajo",  5, 1,  900),
]

DOCUMENT_BUDGET = 50_000
DOCUMENT_N      = 30
DOCUMENT_SEED   = 42

FIXED_GA_PARAMS = dict(
    population_size      = 100,
    crossover_rate       = 0.80,
    mutation_rate        = 0.05,
    elite_size           = 2,
    tournament_k         = 3,
    max_generations      = 500,
    stagnation_limit     = 50,
    semi_heuristic_ratio = 0.20,
    guided_mutation_prob = 0.50,
    repair_prob          = 0.80,
)

CORRIDA_CONFIGS = {
    1: {"label": "Línea base",      "crossover_type": "single_point", "guided_mutation": False, "repair_enabled": False},
    2: {"label": "Cruce uniforme",  "crossover_type": "uniform",      "guided_mutation": False, "repair_enabled": False},
    3: {"label": "Mutación guiada", "crossover_type": "single_point", "guided_mutation": True,  "repair_enabled": False},
    4: {"label": "Con reparación",  "crossover_type": "single_point", "guided_mutation": False, "repair_enabled": True},
}


def generate_document_instance() -> Tuple[List[Doctor], List[Establishment], int]:
    rng = random.Random(DOCUMENT_SEED)
    est_ids = [e.id for e in DOCUMENT_ESTABLISHMENTS]
    doctors = [
        Doctor(id=i, name=f"Médico {i}", preferences=rng.sample(est_ids, 3))
        for i in range(1, DOCUMENT_N + 1)
    ]
    return doctors, [Establishment(**asdict(e)) for e in DOCUMENT_ESTABLISHMENTS], DOCUMENT_BUDGET


# ============================================================
# ALGORITMO GENÉTICO
# ============================================================

@dataclass
class GAConfig:
    population_size:      int   = 100
    crossover_rate:       float = 0.80
    mutation_rate:        float = 0.05
    elite_size:           int   = 2
    tournament_k:         int   = 3
    max_generations:      int   = 500
    stagnation_limit:     int   = 50
    semi_heuristic_ratio: float = 0.20
    seed:                 Optional[int] = DOCUMENT_SEED
    crossover_type:       str   = "single_point"
    guided_mutation:      bool  = False
    guided_mutation_prob: float = 0.50
    repair_enabled:       bool  = False
    repair_prob:          float = 0.80


class MedicalAssignmentGA:
    RISK_W = {"Alto": 500, "Medio": 250, "Bajo": 100}

    def __init__(self, doctors, establishments, budget, config: GAConfig):
        self.doctors        = doctors
        self.establishments = establishments
        self.budget         = budget
        self.config         = config
        self.rng            = random.Random(config.seed)
        self.N              = len(doctors)
        self.M              = len(establishments)
        self.est_id_to_idx: Dict[int, int] = {e.id: i for i, e in enumerate(establishments)}
        self.pref_indices: List[List[int]] = [
            [self.est_id_to_idx[eid] for eid in d.preferences]
            for d in doctors
        ]
        self.score_matrix   = self._build_scores()
        self.population: List[List[int]] = []
        self.best_chromosome = None
        self.best_breakdown: Optional[FitnessBreakdown] = None
        self.history_best:  List[float] = []
        self.history_mean:  List[float] = []
        self.current_gen    = 0
        self.running        = False

    def _build_scores(self):
        m = [[0]*self.M for _ in range(self.N)]
        for di, d in enumerate(self.doctors):
            for rank, eid in enumerate(d.preferences):
                m[di][self.est_id_to_idx[eid]] = [30, 20, 10][rank]
        return m

    def rg(self):
        return self.rng.randrange(self.M)

    def init_pop(self):
        pop  = [[self.rg() for _ in range(self.N)] for _ in range(self.config.population_size)]
        semi = max(1, int(self.config.population_size * self.config.semi_heuristic_ratio))
        for i in range(semi):
            chrom = [
                self.pref_indices[di][0] if self.rng.random() < 0.85 else self.rg()
                for di in range(self.N)
            ]
            pop[i] = chrom
        self.population = pop

    def evaluate(self, ch: List[int]) -> FitnessBreakdown:
        asgn = [0]*self.M
        sat, cost = 0, 0
        for di, ei in enumerate(ch):
            asgn[ei] += 1
            sat  += self.score_matrix[di][ei]
            cost += self.establishments[ei].bonus_cost
        p_cap  = sum(max(0, asgn[i] - e.capacity_max) * 1000 for i, e in enumerate(self.establishments))
        p_epi  = sum(max(0, e.min_quota - asgn[i]) * self.RISK_W[e.risk_level] for i, e in enumerate(self.establishments))
        excess = max(0, cost - self.budget)
        p_bud  = int(0.5 * excess)
        fit    = sat - (p_cap + p_epi + p_bud)
        return FitnessBreakdown(fit, sat, p_cap, p_epi, p_bud, cost, excess,
                                {i: asgn[i] for i in range(self.M)})

    def tournament(self, ev):
        return max(self.rng.sample(ev, k=min(self.config.tournament_k, len(ev))),
                   key=lambda x: x[1].fitness)[0][:]

    def crossover(self, p1, p2):
        if self.N < 2 or self.rng.random() > self.config.crossover_rate:
            return p1[:], p2[:]
        if self.config.crossover_type == "uniform":
            c1, c2 = [], []
            for a, b in zip(p1, p2):
                if self.rng.random() < 0.5: c1.append(a); c2.append(b)
                else:                        c1.append(b); c2.append(a)
            return c1, c2
        pt = self.rng.randint(1, self.N - 1)
        return p1[:pt]+p2[pt:], p2[:pt]+p1[pt:]

    def mutate(self, ch):
        if self.rng.random() < self.config.mutation_rate:
            i = self.rng.randrange(self.N)
            if self.config.guided_mutation and self.rng.random() < self.config.guided_mutation_prob:
                ch[i] = self.rng.choice(self.pref_indices[i])
            else:
                ch[i] = self.rg()

    def repair(self, ch):
        if not self.config.repair_enabled or self.rng.random() > self.config.repair_prob:
            return
        cnt = [0]*self.M
        for ei in ch: cnt[ei] += 1
        for di in range(self.N):
            ei = ch[di]
            if cnt[ei] > self.establishments[ei].capacity_max:
                cands = [i for i, e in enumerate(self.establishments) if cnt[i] < e.capacity_max]
                if not cands: break
                pref  = [i for i in cands if i in self.pref_indices[di]]
                tgt   = self.rng.choice(pref) if pref else self.rng.choice(cands)
                cnt[ei] -= 1; cnt[tgt] += 1; ch[di] = tgt

    def evolve(self):
        self.init_pop()
        best_fit = float("-inf")
        stag     = 0
        self.running = True
        for gen in range(self.config.max_generations):
            if not self.running: break
            self.current_gen = gen + 1
            ev = [(ch, self.evaluate(ch)) for ch in self.population]
            ev.sort(key=lambda x: x[1].fitness, reverse=True)
            cbf = ev[0][1].fitness
            self.history_best.append(cbf)
            self.history_mean.append(statistics.mean(x[1].fitness for x in ev))
            if cbf > best_fit:
                best_fit, self.best_chromosome, self.best_breakdown = cbf, ev[0][0][:], ev[0][1]
                stag = 0
            else:
                stag += 1
            if stag >= self.config.stagnation_limit: break
            new_pop = [ev[i][0][:] for i in range(min(self.config.elite_size, len(ev)))]
            while len(new_pop) < self.config.population_size:
                c1, c2 = self.crossover(self.tournament(ev), self.tournament(ev))
                self.mutate(c1); self.mutate(c2)
                self.repair(c1); self.repair(c2)
                new_pop.append(c1)
                if len(new_pop) < self.config.population_size: new_pop.append(c2)
            self.population = new_pop
        self.running = False
        return self.best_chromosome, self.best_breakdown

    def decode(self, ch):
        out = []
        for di, ei in enumerate(ch):
            d, e = self.doctors[di], self.establishments[ei]
            pref_names = [self.establishments[self.est_id_to_idx[eid]].name for eid in d.preferences]
            out.append({
                "doctor_id":                     d.id,
                "doctor_name":                   d.name,
                "preferred_establishments":      d.preferences,
                "preferred_establishment_names": pref_names,
                "assigned_establishment_id":     e.id,
                "assigned_establishment_name":   e.name,
                "assigned_risk_level":           e.risk_level,
                "assigned_bonus_cost":           e.bonus_cost,
                "satisfaction_points":           self.score_matrix[di][ei],
            })
        return out


# ============================================================
# ESTADO DE UNA CORRIDA INDIVIDUAL
# ============================================================

@dataclass
class CorridaState:
    number: int
    label:  str
    status: str = "idle"
    ga: Optional[MedicalAssignmentGA] = None
    breakdown: Optional[FitnessBreakdown] = None
    decoded: List[Dict] = None
    error_msg: str = ""

    def to_dict(self):
        bd = None
        if self.breakdown:
            b  = self.breakdown
            bd = {"fitness": b.fitness, "satisfaction_total": b.satisfaction_total,
                  "penalty_capacity": b.penalty_capacity,
                  "penalty_epidemiological": b.penalty_epidemiological,
                  "penalty_budget": b.penalty_budget,
                  "total_cost": b.total_cost, "budget_excess": b.budget_excess,
                  "assigned_per_establishment": {str(k): v for k, v in b.assigned_per_establishment.items()}}
        gp = {}
        if self.ga:
            gp = {"current_gen": self.ga.current_gen,
                  "history_best": self.ga.history_best[-300:],
                  "history_mean": self.ga.history_mean[-300:]}
        return {"number": self.number, "label": self.label, "status": self.status,
                "breakdown": bd, "decoded": self.decoded or [], "ga_progress": gp,
                "error_msg": self.error_msg}


# ============================================================
# ESTADO GLOBAL
# ============================================================

class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.doctors, self.establishments, self.budget = generate_document_instance()
        self.current_selected_corrida = 0
        self.corridas: Dict[int, CorridaState] = {
            n: CorridaState(n, CORRIDA_CONFIGS[n]["label"]) for n in range(1, 5)
        }
        self.corridas_running = False
        self.has_run_once = False
        self.current_seed = DOCUMENT_SEED          # semilla activa
        self.best_corrida_num: Optional[int] = None  # corrida ganadora del último batch
        self.run_count = 0                          # cuántos batches se han ejecutado
        self.status_message = "Instancia cargada (N=30, M=8, B=S/50 000). Presiona 'Ejecutar las 4 corridas' para comenzar."
        self.status_type    = "info"

    def _make_config(self, crossover_type="single_point",
                     guided_mutation=False, repair_enabled=False,
                     seed=DOCUMENT_SEED) -> GAConfig:
        return GAConfig(**FIXED_GA_PARAMS, seed=seed,
                        crossover_type=crossover_type,
                        guided_mutation=guided_mutation,
                        repair_enabled=repair_enabled)

    def run_all_corridas(self, use_random_seed=False):
        if self.corridas_running: return
        seed = random.randint(0, 999999) if use_random_seed else DOCUMENT_SEED
        with self.lock:
            self.corridas_running = True
            self.has_run_once = True
            self.current_seed = seed
            self.best_corrida_num = None
            self.run_count += 1
            for n in range(1, 5):
                self.corridas[n] = CorridaState(n, CORRIDA_CONFIGS[n]["label"], status="running")
            seed_label = f"semilla aleatoria {seed}" if use_random_seed else f"semilla fija {seed}"
            self.status_message = f"Ejecutando 4 corridas en paralelo… ({seed_label})"
            self.status_type    = "running"

        threads = []
        for n in range(1, 5):
            cc  = CORRIDA_CONFIGS[n]
            cfg = self._make_config(cc["crossover_type"], cc["guided_mutation"],
                                    cc["repair_enabled"], seed=seed)
            docs, ests, bud = generate_document_instance()
            # Regenerate doctors with current seed so preferences vary on random runs
            rng_inst = random.Random(seed)
            est_ids = [e.id for e in DOCUMENT_ESTABLISHMENTS]
            docs = [Doctor(id=i, name=f"Médico {i}",
                           preferences=rng_inst.sample(est_ids, 3))
                    for i in range(1, DOCUMENT_N + 1)]
            ga  = MedicalAssignmentGA(docs, ests, bud, cfg)
            with self.lock:
                self.corridas[n].ga = ga

            def job(n=n, ga=ga):
                try:
                    ch, bd  = ga.evolve()
                    decoded = ga.decode(ch)
                    with self.lock:
                        self.corridas[n].breakdown = bd
                        self.corridas[n].decoded   = decoded
                        self.corridas[n].status    = "done"
                except Exception as e:
                    with self.lock:
                        self.corridas[n].status    = "error"
                        self.corridas[n].error_msg = str(e)
                finally:
                    with self.lock:
                        if all(self.corridas[k].status in ("done", "error") for k in range(1, 5)):
                            self.corridas_running = False
                            done = [n for n in range(1, 5) if self.corridas[n].breakdown]
                            if done:
                                best_n = max(done, key=lambda n: self.corridas[n].breakdown.fitness)
                                self.current_selected_corrida = best_n
                                self.best_corrida_num = best_n
                            seed_used = self.current_seed
                            seed_label = f"semilla aleatoria {seed_used}" if use_random_seed else f"semilla fija {seed_used}"
                            self.status_message = f"Las 4 corridas finalizaron · {seed_label} · batch #{self.run_count}"
                            self.status_type      = "success"

            t = threading.Thread(target=job, daemon=True)
            threads.append(t)
        for t in threads: t.start()

    def reset_corridas(self):
        with self.lock:
            self.corridas = {n: CorridaState(n, CORRIDA_CONFIGS[n]["label"]) for n in range(1, 5)}
            self.corridas_running = False
            self.has_run_once = False
            self.current_selected_corrida = 0
            self.best_corrida_num = None
            self.run_count = 0
            self.current_seed = DOCUMENT_SEED
            self.status_message = "Reiniciado. Presiona 'Ejecutar las 4 corridas' para comenzar."
            self.status_type = "info"

    def select_corrida(self, corrida_num: int):
        with self.lock:
            if 1 <= corrida_num <= 4:
                self.current_selected_corrida = corrida_num

    def get_state(self):
        with self.lock:
            free_bd = None
            free_decoded = []
            best_info = None
            sel = self.current_selected_corrida
            if 1 <= sel <= 4:
                c = self.corridas[sel]
                if c.breakdown:
                    b = c.breakdown
                    free_bd = {"fitness": b.fitness, "satisfaction_total": b.satisfaction_total,
                               "penalty_capacity": b.penalty_capacity,
                               "penalty_epidemiological": b.penalty_epidemiological,
                               "penalty_budget": b.penalty_budget, "total_cost": b.total_cost,
                               "budget_excess": b.budget_excess,
                               "assigned_per_establishment": {str(k): v for k, v in b.assigned_per_establishment.items()}}
                    free_decoded = c.decoded or []
                    cc = CORRIDA_CONFIGS[sel]
                    best_info = {
                        "corrida_num":    sel,
                        "label":          c.label,
                        "crossover_type": cc["crossover_type"],
                        "guided_mutation": cc["guided_mutation"],
                        "repair_enabled": cc["repair_enabled"],
                        "seed":           self.current_seed,
                        "is_best":        sel == self.best_corrida_num,
                        "run_count":      self.run_count,
                    }
            return {
                "n_doctors":         len(self.doctors),
                "n_establishments":  len(self.establishments),
                "budget":            self.budget,
                "seed":              self.current_seed,
                "corridas_running":  self.corridas_running,
                "has_run_once":      self.has_run_once,
                "run_count":         self.run_count,
                "status_message":    self.status_message,
                "status_type":       self.status_type,
                "establishments":    [asdict(e) for e in self.establishments],
                "free_breakdown":    free_bd,
                "free_decoded":      free_decoded,
                "best_info":         best_info,
                "fixed_params":      FIXED_GA_PARAMS,
                "corridas":          {str(n): self.corridas[n].to_dict() for n in range(1, 5)},
            }


STATE = AppState()


# ============================================================
# FRONTEND HTML
# ============================================================

HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AG SERUMS · MINSA · UNMSM 2026-I</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#060a12;--surf:#0d1520;--surf2:#111d2e;--surf3:#162338;
  --brd:#1c2d45;--brd2:#243a57;
  --txt:#ccd9f5;--txt2:#6a86b8;--txt3:#334d78;
  --acc:#3b7fff;--acc2:#00d4c8;--acc3:#b07ef8;
  --ok:#1fd8a4;--warn:#f5a623;--err:#f05050;
  --c1:#3b7fff;--c2:#f5a623;--c3:#1fd8a4;--c4:#b07ef8;
  --font:'IBM Plex Sans',sans-serif;--mono:'IBM Plex Mono',monospace;
}
html{font-family:var(--font);background:var(--bg);color:var(--txt)}
body{min-height:100vh}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:var(--surf)}::-webkit-scrollbar-thumb{background:var(--brd2);border-radius:2px}
.shell{display:grid;grid-template-rows:auto 1fr;height:100vh;overflow:hidden}
.hdr{padding:12px 22px;border-bottom:1px solid var(--brd);background:rgba(6,10,18,.95);backdrop-filter:blur(16px);position:sticky;top:0;z-index:100}
.hdr-top{display:flex;align-items:center;gap:12px;justify-content:space-between;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:10px}
.brand-badge{background:linear-gradient(135deg,var(--acc),var(--acc2));border-radius:8px;padding:7px 9px;font-size:15px;line-height:1}
.brand-text .name{font-size:15px;font-weight:700;letter-spacing:-.2px}
.brand-text .sub{font-size:10px;color:var(--txt3);margin-top:1px;font-family:var(--mono)}
.hdr-chips{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.chip{background:var(--surf2);border:1px solid var(--brd);border-radius:5px;padding:3px 10px;font-size:10px;color:var(--txt3);font-family:var(--mono);white-space:nowrap}
.chip b{color:var(--acc2)}
.sbar{margin-top:8px;display:flex;align-items:center;gap:8px;font-size:11px;color:var(--txt2)}
.sdot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.sdot.success{background:var(--ok)}.sdot.running{background:var(--warn);animation:pulse 1s infinite}
.sdot.error{background:var(--err)}.sdot.info{background:var(--acc)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.body{display:grid;grid-template-columns:1fr;height:calc(100vh - 88px);overflow:hidden}
.content{overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.btn{display:inline-flex;align-items:center;gap:5px;padding:6px 14px;border-radius:6px;border:none;cursor:pointer;font-family:var(--font);font-size:12px;font-weight:600;transition:all .14s;white-space:nowrap}
.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-ok{background:linear-gradient(135deg,#0fa870,#1fd8a4);color:#fff}
.btn-ok:hover:not(:disabled){filter:brightness(1.1);transform:translateY(-1px)}
.btn-ghost{background:var(--surf3);color:var(--txt2);border:1px solid var(--brd2)}
.btn-ghost:hover:not(:disabled){color:var(--txt);border-color:var(--acc)}
.tabs{display:flex;gap:2px;padding:3px;background:var(--surf);border-radius:7px;flex-wrap:wrap}
.tab{padding:5px 12px;border-radius:5px;border:none;background:none;color:var(--txt3);font-family:var(--font);font-size:11px;font-weight:600;cursor:pointer;transition:all .15s;white-space:nowrap}
.tab.active{background:var(--surf3);color:var(--txt);border:1px solid var(--brd2)}
.tab:hover:not(.active){color:var(--txt2)}
.tc{display:none}.tc.active{display:block}
.card{background:var(--surf2);border:1px solid var(--brd);border-radius:11px;overflow:hidden}
.card-hdr{padding:10px 14px 8px;border-bottom:1px solid var(--brd);display:flex;align-items:center;justify-content:space-between}
.card-title{font-size:12px;font-weight:700;display:flex;align-items:center;gap:6px;color:var(--txt)}
.card-body{padding:12px 14px}
.ch{height:160px;position:relative}
canvas{display:block;border-radius:6px}
.chart-legend{display:flex;gap:14px;margin-top:7px;font-size:10px;color:var(--txt2)}
.chart-legend span{display:flex;align-items:center;gap:5px}
.leg-line{display:inline-block;width:14px;height:2px;vertical-align:middle;border-radius:1px}
.empty{text-align:center;padding:36px 20px;color:var(--txt3)}
.empty .ic{font-size:34px;margin-bottom:8px;opacity:.4}
.empty h3{font-size:13px;font-weight:600;color:var(--txt2);margin-bottom:3px}
.empty p{font-size:11px;line-height:1.6}
.eg{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}
.ec{background:var(--surf3);border:1px solid var(--brd);border-radius:9px;padding:10px}
.ec-id{font-size:9px;font-family:var(--mono);color:var(--txt3);margin-bottom:3px}
.ec-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px}
.ec-name{font-size:12px;font-weight:600}
.bt{height:4px;background:var(--brd);border-radius:2px;margin:5px 0 4px;overflow:hidden}
.bf{height:100%;border-radius:2px;transition:width .6s}
.ec-meta{display:flex;justify-content:space-between;font-size:10px;color:var(--txt2);margin-top:4px}
.ec-st{font-size:9px;font-weight:700;letter-spacing:.3px}
.st-ok{color:var(--ok)}.st-ov{color:var(--err)}.st-sh{color:var(--warn)}
.bdg{display:inline-flex;align-items:center;gap:3px;padding:2px 7px;border-radius:20px;font-size:9px;font-weight:700;white-space:nowrap}
.ba{background:rgba(240,80,80,.12);color:#f05050;border:1px solid rgba(240,80,80,.25)}
.bm{background:rgba(245,166,35,.12);color:#f5a623;border:1px solid rgba(245,166,35,.25)}
.bb{background:rgba(31,216,164,.12);color:#1fd8a4;border:1px solid rgba(31,216,164,.25)}
.b30{background:rgba(31,216,164,.12);color:var(--ok);border:1px solid rgba(31,216,164,.25)}
.b20{background:rgba(245,166,35,.12);color:var(--warn);border:1px solid rgba(245,166,35,.25)}
.b10{background:rgba(240,80,80,.1);color:var(--err);border:1px solid rgba(240,80,80,.2)}
.b0{background:rgba(51,77,120,.2);color:var(--txt3);border:1px solid var(--brd)}
.tw{overflow:auto;max-height:420px;border-radius:8px;border:1px solid var(--brd)}
table{width:100%;border-collapse:collapse;font-size:11px}
thead{position:sticky;top:0;z-index:5}
thead th{background:var(--surf3);padding:7px 10px;text-align:left;font-size:9px;font-weight:700;color:var(--txt3);text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--brd2);white-space:nowrap}
tbody tr{border-bottom:1px solid var(--brd);transition:background .1s}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:var(--surf3)}
tbody td{padding:7px 10px;vertical-align:middle}
.tm{font-family:var(--mono);font-size:10px}
.frow{display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:8px 12px;border-bottom:1px solid var(--brd);background:var(--surf)}
.finp{background:var(--surf3);border:1px solid var(--brd);border-radius:6px;color:var(--txt);font-family:var(--font);font-size:11px;padding:5px 9px;outline:none;min-width:180px;transition:border-color .15s}
.finp:focus{border-color:var(--acc)}
.fsel{background:var(--surf3);border:1px solid var(--brd);border-radius:6px;color:var(--txt);font-family:var(--font);font-size:11px;padding:5px 9px;outline:none;cursor:pointer}
.bdg-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}
.bc{background:var(--surf3);border:1px solid var(--brd);border-radius:8px;padding:10px}
.bl{font-size:9px;color:var(--txt3);font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px}
.bv{font-size:16px;font-weight:700;font-family:var(--mono)}
.formula-box{background:var(--surf3);border:1px solid var(--brd);border-left:3px solid var(--acc);border-radius:8px;padding:12px 14px;font-family:var(--mono);font-size:10px;color:var(--txt2);line-height:2}
.corridas-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.corrida-card{background:var(--surf3);border:1px solid var(--brd);border-radius:10px;overflow:hidden}
.corrida-hdr{padding:9px 12px 7px;border-bottom:1px solid var(--brd);display:flex;align-items:center;justify-content:space-between}
.corrida-label{font-size:11px;font-weight:700;margin-left:7px}
.corrida-factor{font-size:9px;color:var(--txt3);font-family:var(--mono);margin-top:1px}
.cs{font-size:9px;padding:2px 7px;border-radius:20px;font-weight:700}
.cs-idle{background:var(--surf2);color:var(--txt3);border:1px solid var(--brd)}
.cs-running{background:rgba(245,166,35,.12);color:var(--warn);border:1px solid rgba(245,166,35,.25);animation:pulse 1.2s infinite}
.cs-done{background:rgba(31,216,164,.12);color:var(--ok);border:1px solid rgba(31,216,164,.25)}
.cs-error{background:rgba(240,80,80,.1);color:var(--err);border:1px solid rgba(240,80,80,.2)}
.corrida-body{padding:10px 12px}
.corrida-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:8px}
.cm{background:var(--surf2);border:1px solid var(--brd);border-radius:6px;padding:6px 8px}
.cml{font-size:8px;color:var(--txt3);font-weight:700;text-transform:uppercase;letter-spacing:.4px;margin-bottom:2px}
.cmv{font-size:14px;font-weight:700;font-family:var(--mono)}
.corrida-chart{height:70px;position:relative}
.corrida-empty{text-align:center;padding:14px 8px;color:var(--txt3);font-size:10px}
.compare-chart{height:200px;position:relative;margin-bottom:10px}
.cmp-legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--txt2);margin-bottom:8px}
.cmp-legend-item{display:flex;align-items:center;gap:5px}
.cleg-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.overlay{position:fixed;inset:0;background:rgba(6,10,18,.85);backdrop-filter:blur(8px);z-index:200;display:flex;align-items:center;justify-content:center;opacity:0;pointer-events:none;transition:opacity .22s}
.overlay.on{opacity:1;pointer-events:all}
.ov-box{background:var(--surf2);border:1px solid var(--brd2);border-radius:14px;padding:28px 36px;text-align:center;max-width:380px}
.spin{width:44px;height:44px;border:3px solid var(--brd2);border-top-color:var(--acc);border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 14px}
@keyframes spin{to{transform:rotate(360deg)}}
/* Pref tags */
.pt{font-size:9px;padding:2px 6px;border-radius:3px;background:var(--surf);border:1px solid var(--brd);color:var(--txt3);white-space:nowrap;display:inline-block}
.pt.p1{border-color:var(--ok);color:var(--ok)}
.pt.p2{border-color:var(--warn);color:var(--warn)}
.pt.p3{border-color:var(--err);color:var(--err)}
.fact-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.fact-row{background:var(--surf3);border:1px solid var(--brd);border-radius:7px;padding:7px 9px;display:flex;align-items:center;gap:7px}
.fact-text{font-size:10px;line-height:1.5}
.fact-lbl{color:var(--txt3);font-size:9px}
.fact-val{font-family:var(--mono);font-weight:600;color:var(--ok)}
/* Pantalla de bienvenida */
.welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 20px;text-align:center;gap:20px}
.welcome-icon{font-size:56px;opacity:.7}
.welcome-title{font-size:22px;font-weight:700;color:var(--txt)}
.welcome-sub{font-size:13px;color:var(--txt2);line-height:1.7;max-width:520px}
.welcome-btn{padding:12px 28px;font-size:14px;border-radius:8px;background:linear-gradient(135deg,#0fa870,#1fd8a4);color:#fff;border:none;cursor:pointer;font-family:var(--font);font-weight:700;transition:all .16s;display:inline-flex;align-items:center;gap:8px}
.welcome-btn:hover{filter:brightness(1.12);transform:translateY(-2px);box-shadow:0 8px 24px rgba(31,216,164,.25)}
.welcome-chips{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
.welcome-chip{background:var(--surf3);border:1px solid var(--brd2);border-radius:20px;padding:4px 14px;font-size:11px;color:var(--txt2);font-family:var(--mono)}
/* Banner de corrida ganadora */
.best-banner{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:9px 14px;border-radius:8px;border:1px solid rgba(31,216,164,.3);background:rgba(31,216,164,.06);font-size:11px;color:var(--txt2);line-height:1.6}
.best-banner .bb-icon{font-size:15px;flex-shrink:0}
.best-banner .bb-title{font-weight:700;color:var(--ok);margin-right:4px}
.best-banner .bb-tag{display:inline-flex;align-items:center;gap:3px;padding:1px 8px;border-radius:20px;font-size:9px;font-weight:700;background:rgba(0,212,200,.1);border:1px solid rgba(0,212,200,.25);color:var(--acc2);font-family:var(--mono);white-space:nowrap}
.best-banner .bb-tag.rnd{background:rgba(176,126,248,.1);border-color:rgba(176,126,248,.3);color:var(--acc3)}
.rc-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;border-radius:20px;font-size:10px;font-weight:700;background:rgba(59,127,255,.1);border:1px solid rgba(59,127,255,.25);color:var(--acc);font-family:var(--mono)}
</style>
</head>
<body>
<div class="shell">

<header class="hdr">
  <div class="hdr-top">
    <div class="brand">
      <div class="brand-badge">🏥</div>
      <div class="brand-text">
        <div class="name">AG SERUMS — Asignación Médica Rural</div>
        <div class="sub">MINSA · Grupo 6 · Sistemas Inteligentes 2026-I · UNMSM · Fac. Ingeniería de Sistemas</div>
      </div>
    </div>
    <div class="hdr-chips">
      <span class="chip">N = <b>30</b> médicos</span>
      <span class="chip">M = <b>8</b> establecimientos</span>
      <span class="chip">B = <b>S/ 50 000</b></span>
      <span class="chip">seed = <b>42</b></span>
    </div>
  </div>
  <div class="sbar">
    <div class="sdot info" id="sdot"></div>
    <span id="smsg" style="color:var(--txt2)">Cargando…</span>
  </div>
</header>

<div class="body">
<main class="content">

  <!-- PANTALLA DE BIENVENIDA (visible hasta primera ejecución) -->
  <div class="card" id="welcome-screen">
    <div class="card-body">
      <div class="welcome">
        <div class="welcome-icon">🧬</div>
        <div class="welcome-title">Optimización por Algoritmo Genético</div>
        <div class="welcome-sub">
          Se ejecutarán <strong style="color:var(--acc2)">4 corridas en paralelo</strong>, cada una variando un único factor experimental sobre la misma instancia (N=30 médicos, M=8 establecimientos, semilla=42).
        </div>
        <div class="welcome-chips">
          <span class="welcome-chip">C1 · Línea base</span>
          <span class="welcome-chip">C2 · Cruce uniforme</span>
          <span class="welcome-chip">C3 · Mutación guiada</span>
          <span class="welcome-chip">C4 · Con reparación</span>
        </div>
        <button class="welcome-btn" id="btn-welcome" onclick="runCorridas(false)">▶ Ejecutar las 4 corridas</button>
      </div>
    </div>
  </div>

  <!-- CONTENIDO PRINCIPAL (oculto hasta primera ejecución) -->
  <div id="main-content" style="display:none;flex-direction:column;gap:12px">

    <div class="card" style="flex-shrink:0">
      <div class="card-body" style="padding:6px 10px">
        <div class="tabs">
          <button class="tab active" onclick="switchTab('corridas')">🧪 4 Corridas</button>
          <button class="tab" onclick="switchTab('establecimientos')">🏨 Establecimientos</button>
          <button class="tab" onclick="switchTab('asignaciones')">👨‍⚕️ Asignaciones</button>
          <button class="tab" onclick="switchTab('desglose')">📊 Desglose</button>
        </div>
      </div>
    </div>

    <!-- 4 CORRIDAS -->
    <div class="tc active card" id="tc-corridas">
      <div class="card-hdr">
        <div class="card-title">🧪 4 Corridas Experimentales en Paralelo</div>
        <div style="display:flex;gap:7px;align-items:center;flex-wrap:wrap">
          <button class="btn btn-ok" id="btn-corridas-rnd" onclick="runCorridas(true)">🎲 Nueva corrida</button>
          <button class="btn btn-ghost" onclick="resetCorridas()">🗑 Reiniciar</button>
        </div>
      </div>
      <div class="card-body">

        <div class="card" style="margin-bottom:12px">
          <div class="card-hdr"><div class="card-title">📈 Convergencia comparativa</div></div>
          <div class="card-body">
            <div class="cmp-legend">
              <div class="cmp-legend-item"><div class="cleg-dot" style="background:var(--c1)"></div><span>C1 · Línea base</span></div>
              <div class="cmp-legend-item"><div class="cleg-dot" style="background:var(--c2)"></div><span>C2 · Cruce uniforme</span></div>
              <div class="cmp-legend-item"><div class="cleg-dot" style="background:var(--c3)"></div><span>C3 · Mutación guiada</span></div>
              <div class="cmp-legend-item"><div class="cleg-dot" style="background:var(--c4)"></div><span>C4 · Con reparación</span></div>
            </div>
            <div class="compare-chart"><canvas id="cvs-compare"></canvas></div>
          </div>
        </div>
        <div class="tw" style="margin-bottom:12px;max-height:145px">
          <table><thead><tr>
            <th>Corrida</th><th>Factor</th><th>Hipótesis</th><th>Estado</th>
            <th>Fitness</th><th>Satisf.</th><th>P.Cap</th><th>P.Epi</th><th>P.Ppto</th><th>Costo</th><th>Gen.</th>
          </tr></thead>
          <tbody id="cr-tbody"></tbody></table>
        </div>
        <div class="corridas-grid" id="corridas-grid"></div>
      </div>
    </div>

    <!-- ESTABLECIMIENTOS -->
    <div class="tc card" id="tc-establecimientos">
      <div class="card-hdr"><div class="card-title">🏨 Establecimientos</div></div>
      <div class="card-body">
        <div id="banner-est" style="display:none;margin-bottom:12px"></div>
        <div class="fact-grid" style="margin-bottom:12px">
          <div class="fact-row"><span style="font-size:13px">✅</span><div class="fact-text"><div class="fact-lbl">Condición 1 — Capacidad ≥ N</div><div class="fact-val">ΣC = 38 ≥ 30 ✓</div></div></div>
          <div class="fact-row"><span style="font-size:13px">✅</span><div class="fact-text"><div class="fact-lbl">Condición 2 — Cuotas ≤ N</div><div class="fact-val">ΣQ = 19 ≤ 30 ✓</div></div></div>
          <div class="fact-row"><span style="font-size:13px">✅</span><div class="fact-text"><div class="fact-lbl">Condición 3 — Q_j ≤ C_j</div><div class="fact-val">8/8 centros ✓</div></div></div>
          <div class="fact-row"><span style="font-size:13px">✅</span><div class="fact-text"><div class="fact-lbl">Condición 4 — K_mín ≤ B</div><div class="fact-val">48 000 ≤ 50 000 ✓</div></div></div>
        </div>
        <div class="eg" id="est-grid"></div>
      </div>
    </div>

    <!-- ASIGNACIONES -->
    <div class="tc card" id="tc-asignaciones">
      <div class="card-hdr">
        <div class="card-title">👨‍⚕️ Asignación Individual</div>
        <span id="a-cnt" style="font-size:10px;color:var(--txt3)">0 asig.</span>
      </div>
      <div id="banner-asig" style="display:none;padding:0 12px;margin-top:10px;margin-bottom:2px"></div>
      <div class="frow">
        <input class="finp" id="flt-n" placeholder="🔍 Médico o establecimiento…" oninput="applyF()">
        <select class="fsel" id="flt-r" onchange="applyF()">
          <option value="">Todos los riesgos</option>
          <option value="Alto">🔴 Alto</option><option value="Medio">🟡 Medio</option><option value="Bajo">🟢 Bajo</option>
        </select>
        <select class="fsel" id="flt-p" onchange="applyF()">
          <option value="">Todos los puntos</option>
          <option value="30">30 — 1ª pref.</option><option value="20">20 — 2ª pref.</option>
          <option value="10">10 — 3ª pref.</option><option value="0">0 — fuera</option>
        </select>
        <span id="flt-cnt" style="font-size:10px;color:var(--txt3);margin-left:auto"></span>
      </div>
      <div class="tw" id="tbl-wrap">
        <div class="empty"><div class="ic">🩺</div><h3>Sin asignaciones</h3><p>Ejecuta las corridas primero.</p></div>
      </div>
    </div>

    <!-- DESGLOSE -->
    <div class="tc card" id="tc-desglose">
      <div class="card-hdr"><div class="card-title">📊 Desglose del Fitness</div></div>
      <div class="card-body">
        <div id="banner-desglose" style="display:none;margin-bottom:12px"></div>
        <div class="empty" id="bd-empty"><div class="ic">📐</div><h3>Sin resultados</h3><p>Ejecuta las corridas primero.</p></div>
        <div id="bd-content" style="display:none">
          <div class="bdg-grid">
            <div class="bc"><div class="bl">FITNESS</div><div class="bv ca" id="bd-fit">—</div></div>
            <div class="bc"><div class="bl">Satisfacción</div><div class="bv co" id="bd-sat">—</div></div>
            <div class="bc"><div class="bl">Pen. Capacidad</div><div class="bv ce" id="bd-pc">—</div></div>
            <div class="bc"><div class="bl">Pen. Epidemiol.</div><div class="bv ce" id="bd-pe">—</div></div>
            <div class="bc"><div class="bl">Pen. Presupuesto</div><div class="bv ce" id="bd-pb">—</div></div>
            <div class="bc"><div class="bl">Costo total</div><div class="bv cw" id="bd-ct">—</div></div>
            <div class="bc"><div class="bl">Presupuesto B</div><div class="bv cc">S/ 50 000</div></div>
            <div class="bc"><div class="bl">Exceso</div><div class="bv" id="bd-ex">—</div></div>
          </div>
          <div style="margin:10px 0 12px">
            <div style="font-size:10px;font-weight:700;margin-bottom:6px;color:var(--txt2)">Distribución de satisfacción — por preferencia de establecimiento (N=30)</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap" id="sat-dist"></div>
          </div>
          <div class="formula-box">
            FITNESS(x) = S(x) − [ P<sub>cap</sub>(x) + P<sub>epi</sub>(x) + P<sub>pre</sub>(x) ]<br>
            <span style="color:var(--acc2)">S(x) = Σ s(i)  ·  +30 (1ª pref. estab.)  ·  +20 (2ª)  ·  +10 (3ª)  ·  0 (sin pref.)</span><br>
            <span style="color:var(--err)">P<sub>cap</sub> = 1000 · Σ max(0, n_j − C_j)           — penalización por exceso de capacidad</span><br>
            <span style="color:var(--err)">P<sub>epi</sub> = Σ w_j · max(0, Q_j − n_j)  w={Alto:500, Medio:250, Bajo:100}  — penalización epidemiológica</span><br>
            <span style="color:var(--warn)">P<sub>pre</sub> = 0.5 · max(0, K(x) − B)              — penalización por exceso de presupuesto</span>
          </div>
        </div>
      </div>
    </div>

  </div><!-- /main-content -->
</main>
</div>
</div>

<div class="overlay" id="overlay">
  <div class="ov-box">
    <div class="spin"></div>
    <div style="font-size:16px;font-weight:700;margin-bottom:4px">Optimizando…</div>
    <div style="font-size:11px;color:var(--txt2)">Ejecutando 4 corridas en paralelo…</div>
  </div>
</div>

<script>
let S={},allSol=[],pollInt=null;
let charts={compare:null};
const COLORS={1:'#3b7fff',2:'#f5a623',3:'#1fd8a4',4:'#b07ef8'};
const FACTORS={1:'Referencia',2:'Tipo cruzamiento (M4)',3:'Mutación por estab. preferido (M5)',4:'Manejo restricciones (M6)'};
const HYP={1:'—',2:'Mejor mezcla',3:'Mayor satisfacción',4:'Converge antes'};

function mkChart(id){
  const c=document.getElementById(id);if(!c)return null;
  const ch={canvas:c,ctx:c.getContext('2d'),data:null};
  const r=()=>{c.width=c.parentElement.offsetWidth;c.height=c.parentElement.offsetHeight;if(ch.data)drawSingle(ch);};
  ch.resize=r;r();window.addEventListener('resize',r);return ch;
}
function drawSingle(ch,color='#3b7fff'){
  const{canvas:c,ctx,data}=ch;if(!data)return;
  const W=c.width,H=c.height,P={t:14,r:10,b:28,l:48};
  ctx.clearRect(0,0,W,H);
  const best=data.best||[],mean=data.mean||[];
  if(best.length<2)return;
  const all=[...best,...mean],mn=Math.min(...all),mx=Math.max(...all),rng=mx-mn||1;
  const tx=i=>P.l+(i/(best.length-1))*(W-P.l-P.r);
  const ty=v=>P.t+(1-(v-mn)/rng)*(H-P.t-P.b);
  ctx.strokeStyle='#1c2d45';ctx.lineWidth=1;
  for(let i=0;i<=3;i++){const y=P.t+i*(H-P.t-P.b)/3;ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(W-P.r,y);ctx.stroke();ctx.fillStyle='#334d78';ctx.font='9px IBM Plex Mono,monospace';ctx.textAlign='right';ctx.fillText((mx-i*rng/3).toFixed(0),P.l-3,y+3);}
  const steps=Math.min(6,best.length-1);ctx.fillStyle='#334d78';ctx.font='9px sans-serif';ctx.textAlign='center';
  for(let i=0;i<=steps;i++){const idx=Math.round(i*(best.length-1)/steps);ctx.fillText(idx,tx(idx),H-P.b+12);}
  if(mean.length>=2){ctx.beginPath();ctx.strokeStyle='rgba(0,212,200,.4)';ctx.lineWidth=1.2;mean.forEach((v,i)=>{i===0?ctx.moveTo(tx(i),ty(v)):ctx.lineTo(tx(i),ty(v));});ctx.stroke();}
  ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2;best.forEach((v,i)=>{i===0?ctx.moveTo(tx(i),ty(v)):ctx.lineTo(tx(i),ty(v));});ctx.stroke();
  ctx.beginPath();best.forEach((v,i)=>{i===0?ctx.moveTo(tx(i),ty(v)):ctx.lineTo(tx(i),ty(v));});
  ctx.lineTo(tx(best.length-1),H-P.b);ctx.lineTo(tx(0),H-P.b);ctx.closePath();
  const fg=ctx.createLinearGradient(0,P.t,0,H-P.b);fg.addColorStop(0,color+'22');fg.addColorStop(1,'rgba(0,0,0,0)');ctx.fillStyle=fg;ctx.fill();
  const lx=tx(best.length-1),ly=ty(best[best.length-1]);
  ctx.beginPath();ctx.arc(lx,ly,3,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();
}
function drawCompare(cmpData){
  const ch=charts.compare;if(!ch)return;
  const{canvas:c,ctx}=ch;
  const W=c.width,H=c.height,P={t:14,r:10,b:28,l:48};
  ctx.clearRect(0,0,W,H);
  const series=Object.entries(cmpData).map(([n,d])=>({n:parseInt(n),data:d})).filter(s=>s.data&&s.data.length>=2);
  if(!series.length)return;
  const all=series.flatMap(s=>s.data),mn=Math.min(...all),mx=Math.max(...all),rng=mx-mn||1;
  const maxLen=Math.max(...series.map(s=>s.data.length));
  const tx=i=>P.l+(i/(maxLen-1))*(W-P.l-P.r);
  const ty=v=>P.t+(1-(v-mn)/rng)*(H-P.t-P.b);
  ctx.strokeStyle='#1c2d45';ctx.lineWidth=1;
  for(let i=0;i<=3;i++){const y=P.t+i*(H-P.t-P.b)/3;ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(W-P.r,y);ctx.stroke();ctx.fillStyle='#334d78';ctx.font='9px monospace';ctx.textAlign='right';ctx.fillText((mx-i*rng/3).toFixed(0),P.l-3,y+3);}
  const steps=Math.min(8,maxLen-1);ctx.fillStyle='#334d78';ctx.font='9px sans-serif';ctx.textAlign='center';
  for(let i=0;i<=steps;i++){const idx=Math.round(i*(maxLen-1)/steps);ctx.fillText(idx,tx(idx),H-P.b+12);}
  series.forEach(s=>{
    const color=COLORS[s.n];
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2;
    s.data.forEach((v,i)=>{const px=P.l+(i/(maxLen-1))*(W-P.l-P.r);i===0?ctx.moveTo(px,ty(v)):ctx.lineTo(px,ty(v));});
    ctx.stroke();
    const lx=P.l+((s.data.length-1)/(maxLen-1))*(W-P.l-P.r),ly=ty(s.data[s.data.length-1]);
    ctx.beginPath();ctx.arc(lx,ly,3,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();
  });
}
function switchTab(t){
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tc').forEach(c=>c.classList.remove('active'));
  const order=['corridas','establecimientos','asignaciones','desglose'];
  document.querySelectorAll('.tab')[order.indexOf(t)].classList.add('active');
  document.getElementById('tc-'+t).classList.add('active');
}
function fm(v){return'S/ '+Number(v).toLocaleString('es-PE');}
function rb(r){const m={'Alto':'ba','Medio':'bm','Bajo':'bb'},d={'Alto':'🔴','Medio':'🟡','Bajo':'🟢'};return`<span class="bdg ${m[r]||''}">${d[r]||''} ${r}</span>`;}
function pb(p){const m={30:'b30',20:'b20',10:'b10',0:'b0'},l={30:'30·1ª',20:'20·2ª',10:'10·3ª',0:'0·—'};return`<span class="bdg ${m[p]||'b0'}">${l[p]||'—'}</span>`;}

function showMainContent(){
  document.getElementById('welcome-screen').style.display='none';
  const mc=document.getElementById('main-content');
  mc.style.display='flex';
  // Init compare chart after it becomes visible
  if(!charts.compare)charts.compare=mkChart('cvs-compare');
}

function renderEst(s){
  const asgn=s.free_breakdown?s.free_breakdown.assigned_per_establishment:{};
  const grid=document.getElementById('est-grid');if(!grid)return;
  grid.innerHTML=s.establishments.map((e,i)=>{
    const cnt=asgn[String(i)]||0,ratio=e.capacity_max>0?Math.min(cnt/e.capacity_max,1):0;
    let sc='st-ok',st='✓ OK',bc='var(--ok)';
    if(cnt>e.capacity_max){sc='st-ov';st='⚠ Exceso';bc='var(--err)';}
    else if(cnt<e.min_quota){sc='st-sh';st='⚠ Cuota';bc='var(--warn)';}
    return`<div class="ec">
      <div class="ec-id">ID ${e.id}</div>
      <div class="ec-top"><div class="ec-name">${e.name}</div>${rb(e.risk_level)}</div>
      <div class="bt"><div class="bf" style="width:${(ratio*100).toFixed(1)}%;background:${bc}"></div></div>
      <div class="ec-meta"><span><b style="color:var(--txt)">${cnt}</b>/${e.capacity_max} asig.</span><span>mín ${e.min_quota}</span></div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:5px">
        <span style="font-size:9px;color:var(--txt3);font-family:var(--mono)">${fm(e.bonus_cost)}/mes</span>
        <span class="ec-st ${sc}">${st}</span>
      </div>
    </div>`;
  }).join('');
}
function renderAssign(s){
  allSol=s.free_decoded||[];
  const cnt=document.getElementById('a-cnt');if(cnt)cnt.textContent=`${allSol.length} asig.`;
  applyF();
}
function applyF(){
  const nq=(document.getElementById('flt-n')||{}).value||'',rq=(document.getElementById('flt-r')||{}).value||'',pq=(document.getElementById('flt-p')||{}).value||'';
  const f=allSol.filter(x=>(!nq||(x.doctor_name+x.assigned_establishment_name).toLowerCase().includes(nq.toLowerCase()))&&(!rq||x.assigned_risk_level===rq)&&(!pq||String(x.satisfaction_points)===pq));
  const fc=document.getElementById('flt-cnt');if(fc)fc.textContent=f.length?`${f.length} resultados`:'';
  const w=document.getElementById('tbl-wrap');if(!w)return;
  if(!allSol.length){w.innerHTML='<div class="empty"><div class="ic">🩺</div><h3>Sin asignaciones</h3><p>Ejecuta las corridas primero.</p></div>';return;}
  if(!f.length){w.innerHTML='<div class="empty"><div class="ic">🔍</div><h3>Sin resultados</h3><p>Ajusta los filtros.</p></div>';return;}
  w.innerHTML=`<table><thead><tr>
    <th>#</th><th>Médico</th>
    <th>1ª Pref. (estab.)</th><th>2ª Pref. (estab.)</th><th>3ª Pref. (estab.)</th>
    <th>Establecimiento asignado</th><th>Riesgo</th><th>Bono</th><th>Pts.</th>
  </tr></thead><tbody>${f.map(x=>`<tr>
    <td class="tm" style="color:var(--txt3)">${x.doctor_id}</td>
    <td style="font-weight:600">${x.doctor_name}</td>
    <td><span class="pt p1">${x.preferred_establishment_names[0]}</span></td>
    <td><span class="pt p2">${x.preferred_establishment_names[1]}</span></td>
    <td><span class="pt p3">${x.preferred_establishment_names[2]}</span></td>
    <td style="font-weight:600">${x.assigned_establishment_name}</td>
    <td>${rb(x.assigned_risk_level)}</td>
    <td class="tm">${fm(x.assigned_bonus_cost)}</td>
    <td>${pb(x.satisfaction_points)}</td>
  </tr>`).join('')}</tbody></table>`;
}
function renderDesglose(s){
  const em=document.getElementById('bd-empty'),ct=document.getElementById('bd-content');
  if(!em||!ct)return;
  if(!s.free_breakdown){em.style.display='block';ct.style.display='none';return;}
  em.style.display='none';ct.style.display='block';
  const b=s.free_breakdown;
  document.getElementById('bd-fit').textContent=b.fitness.toFixed(0);
  document.getElementById('bd-sat').textContent=b.satisfaction_total;
  document.getElementById('bd-pc').textContent=`-${b.penalty_capacity}`;
  document.getElementById('bd-pe').textContent=`-${b.penalty_epidemiological}`;
  document.getElementById('bd-pb').textContent=`-${b.penalty_budget}`;
  document.getElementById('bd-ct').textContent=fm(b.total_cost);
  const ex=b.budget_excess,el=document.getElementById('bd-ex');
  el.textContent=ex>0?fm(ex):'Sin exceso ✓';el.style.color=ex>0?'var(--err)':'var(--ok)';
  const sol=s.free_decoded||[],dist={30:0,20:0,10:0,0:0};
  sol.forEach(x=>{dist[x.satisfaction_points]=(dist[x.satisfaction_points]||0)+1;});
  const tot=sol.length||1;
  document.getElementById('sat-dist').innerHTML=
    [[30,'var(--ok)','1ª pref. estab.'],[20,'var(--warn)','2ª pref. estab.'],[10,'var(--err)','3ª pref. estab.'],[0,'var(--txt3)','Sin preferencia']].map(([p,c,l])=>`
    <div style="background:var(--surf3);border:1px solid var(--brd);border-radius:7px;padding:8px 11px;min-width:115px">
      <div style="font-size:17px;font-weight:700;color:${c};font-family:var(--mono)">${dist[p]}</div>
      <div style="font-size:9px;color:var(--txt3);margin-top:1px">${l}</div>
      <div style="font-size:10px;color:${c};margin-top:2px">${((dist[p]/tot)*100).toFixed(1)}%</div>
    </div>`).join('');
}
function renderCorridas(s){
  const crs=s.corridas;
  const tb=document.getElementById('cr-tbody');if(!tb)return;
  tb.innerHTML=[1,2,3,4].map(n=>{
    const c=crs[String(n)],color=COLORS[n];
    const stCls={'idle':'','running':'cw','done':'co','error':'ce'}[c.status]||'';
    const stTxt={'idle':'⏳ esperando','running':'⚙ ejecutando…','done':'✓ listo','error':'✗ error'}[c.status]||'';
    if(!c.breakdown)return`<tr><td class="tm"><span style="color:${color}">C${n}</span></td><td>${FACTORS[n]}</td><td style="color:var(--txt3)">${HYP[n]}</td><td class="${stCls}">${stTxt}</td><td colspan="7" style="color:var(--txt3)">${c.ga_progress?.current_gen?'gen '+c.ga_progress.current_gen:'—'}</td></tr>`;
    const b=c.breakdown;
    return`<tr><td class="tm"><span style="color:${color}">C${n}</span></td><td>${FACTORS[n]}</td><td style="color:var(--txt3)">${HYP[n]}</td><td class="${stCls}">${stTxt}</td>
    <td class="tm" style="color:${color}"><b>${b.fitness.toFixed(0)}</b></td><td class="tm">${b.satisfaction_total}</td>
    <td class="tm ce">${b.penalty_capacity}</td><td class="tm ce">${b.penalty_epidemiological}</td><td class="tm ce">${b.penalty_budget}</td>
    <td class="tm">${fm(b.total_cost)}</td><td class="tm">${c.ga_progress?.current_gen||'—'}</td></tr>`;
  }).join('');
  const cmpData={};
  [1,2,3,4].forEach(n=>{const c=crs[String(n)];if(c.ga_progress?.history_best?.length>=2)cmpData[n]=c.ga_progress.history_best;});
  if(Object.keys(cmpData).length&&charts.compare)drawCompare(cmpData);
  const confTxt={
    1:'Cruce: un punto · Mutación: aleatoria · Sin reparación',
    2:`<b style="color:var(--c2)">Cruce: uniforme (M4)</b> · Mutación: aleatoria · Sin reparación`,
    3:`Cruce: un punto · <b style="color:var(--c3)">Mutación: guiada por estab. preferido (M5)</b> · Sin reparación`,
    4:`Cruce: un punto · Mutación: aleatoria · <b style="color:var(--c4)">Reparación activa (M6)</b>`,
  };
  const cg=document.getElementById('corridas-grid');if(!cg)return;
  cg.innerHTML=[1,2,3,4].map(n=>{
    const c=crs[String(n)],color=COLORS[n];
    const stCls={'idle':'cs-idle','running':'cs-running','done':'cs-done','error':'cs-error'}[c.status]||'cs-idle';
    const stTxt={'idle':'Esperando','running':'Ejecutando…','done':'Completado','error':'Error'}[c.status]||'—';
    let mH='<div class="corrida-empty">Sin resultados aún</div>';
    if(c.breakdown){const b=c.breakdown;mH=`<div class="corrida-metrics">
      <div class="cm"><div class="cml">Fitness</div><div class="cmv" style="color:${color}">${b.fitness.toFixed(0)}</div></div>
      <div class="cm"><div class="cml">Satisf.</div><div class="cmv co">${b.satisfaction_total}</div></div>
      <div class="cm"><div class="cml">Gen.</div><div class="cmv">${c.ga_progress?.current_gen||'—'}</div></div>
      <div class="cm"><div class="cml">P.Cap</div><div class="cmv ce" style="font-size:12px">${b.penalty_capacity}</div></div>
      <div class="cm"><div class="cml">P.Epi</div><div class="cmv ce" style="font-size:12px">${b.penalty_epidemiological}</div></div>
      <div class="cm"><div class="cml">Costo</div><div class="cmv cw" style="font-size:10px">${fm(b.total_cost)}</div></div>
    </div>`;}
    const mini=c.ga_progress?.history_best?.length>=2?`<div class="corrida-chart"><canvas id="mini-${n}"></canvas></div>`:`<div class="corrida-empty">${c.status==='idle'?'Esperando…':c.status==='running'?'Calculando…':''}</div>`;
    const clickable=c.breakdown?` style="cursor:pointer" onclick="selectCorrida(${n})":`:'';
    return`<div class="corrida-card"${clickable}>
      <div class="corrida-hdr"><div style="display:flex;align-items:center"><div class="num cn${n}" style="width:20px;height:20px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;background:rgba(59,127,255,.2);color:var(--c${n})">${n}</div><div><div class="corrida-label">${c.label}</div><div class="corrida-factor">${FACTORS[n]}</div></div></div><span class="cs ${stCls}">${stTxt}</span></div>
      <div class="corrida-body"><div style="font-size:9px;color:var(--txt3);margin-bottom:8px;line-height:1.8">${confTxt[n]}</div>${mH}${mini}</div>
    </div>`;
  }).join('');
  [1,2,3,4].forEach(n=>{
    const c=crs[String(n)];
    if(c.ga_progress?.history_best?.length>=2){const mc=mkChart(`mini-${n}`);if(mc){mc.data={best:c.ga_progress.history_best,mean:[]};drawSingle(mc,COLORS[n]);}}
  });
}
function buildBanner(info){
  if(!info)return'';
  const crossMap={'single_point':'Un punto','uniform':'Uniforme'};
  const cross=crossMap[info.crossover_type]||info.crossover_type;
  const guided=info.guided_mutation?'<span class="bb-tag">Mutación guiada ✓</span>':'';
  const repair=info.repair_enabled?'<span class="bb-tag">Reparación ✓</span>':'';
  const seedTag=`<span class="bb-tag ${info.seed!==42?'rnd':''}">seed=${info.seed}</span>`;
  const batchTag=`<span class="rc-badge">batch #${info.run_count}</span>`;
  const star=info.is_best?'🏆':'📌';
  return`<div class="best-banner"><span class="bb-icon">${star}</span>
    <div><span class="bb-title">C${info.corrida_num} · ${info.label}</span>
    Mostrando la corrida con <b>mayor fitness</b> del último batch.
    Cruce: <b>${cross}</b> ${guided}${repair} ${seedTag} ${batchTag}</div></div>`;
}
function renderAll(s){
  S=s;
  if(s.has_run_once){
    showMainContent();
    renderEst(s);renderAssign(s);renderDesglose(s);renderCorridas(s);
    // Banners en pestañas secundarias
    const banner=buildBanner(s.best_info);
    ['banner-est','banner-asig','banner-desglose'].forEach(id=>{
      const el=document.getElementById(id);if(!el)return;
      el.innerHTML=banner;el.style.display=banner?'block':'none';
    });
  }
  document.getElementById('smsg').textContent=s.status_message;
  document.getElementById('sdot').className='sdot '+(s.status_type||'info');
  document.getElementById('overlay').classList.toggle('on',s.corridas_running);
  // Deshabilitar todos los botones de ejecución mientras corre
  ['btn-corridas-rnd','btn-welcome'].forEach(id=>{
    const b=document.getElementById(id);if(b)b.disabled=s.corridas_running;
  });
}
async function fetchState(){try{renderAll(await(await fetch('/api/state')).json());}catch(e){}}
async function runCorridas(useRandom){
  ['btn-welcome','btn-corridas-rnd'].forEach(id=>{
    const b=document.getElementById(id);if(b)b.disabled=true;
  });
  await fetch(`/api/run_corridas?random=${useRandom?1:0}`,{method:'POST'});
  startPoll();
}
async function resetCorridas(){
  await fetch('/api/reset_corridas',{method:'POST'});
  // Show welcome screen again
  document.getElementById('welcome-screen').style.display='block';
  document.getElementById('main-content').style.display='none';
  charts.compare=null;
  await fetchState();
}
async function selectCorrida(n){await fetch(`/api/select_corrida?num=${n}`,{method:'POST'});await fetchState();}
function startPoll(){
  if(pollInt)clearInterval(pollInt);
  pollInt=setInterval(async()=>{await fetchState();if(!S.corridas_running){clearInterval(pollInt);pollInt=null;}},700);
}
window.addEventListener('load',async()=>{
  await fetchState();startPoll();
});
window.addEventListener('resize',()=>{charts.compare&&charts.compare.resize();});
</script>
</body>
</html>
"""


# ============================================================
# HTTP SERVER
# ============================================================

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers();self.wfile.write(body)

    def send_text(self, text, mime="text/plain; charset=utf-8", status=200):
        body=text.encode()
        self.send_response(status)
        self.send_header("Content-Type",mime)
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers();self.wfile.write(body)

    def do_GET(self):
        path=urllib.parse.urlparse(self.path).path
        if path in ("/","/index.html"):self.send_text(HTML,mime="text/html; charset=utf-8")
        elif path=="/api/state":self.send_json(STATE.get_state())
        else:self.send_response(404);self.end_headers()

    def do_POST(self):
        parsed=urllib.parse.urlparse(self.path)
        params=urllib.parse.parse_qs(parsed.query)
        if parsed.path=="/api/run_corridas":
            use_random = params.get("random",["0"])[0] == "1"
            STATE.run_all_corridas(use_random_seed=use_random);self.send_json({"ok":True})
        elif parsed.path=="/api/reset_corridas":
            STATE.reset_corridas();self.send_json({"ok":True})
        elif parsed.path=="/api/select_corrida":
            STATE.select_corrida(int(params.get("num",["0"])[0]));self.send_json({"ok":True})
        else:self.send_response(404);self.end_headers()


def main():
    port = int(os.environ.get("PORT", 8765))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\n{'='*62}")
    print(f"  AG SERUMS — Asignación de Médicos Rurales · MINSA")
    print(f"  Grupo 6 · Sistemas Inteligentes 2026-I · UNMSM")
    print(f"{'='*62}")
    print(f"  Puerto : {port}")
    print(f"  Instancia : N=30, M=8, B=S/50 000")
    print(f"  Preferencias: sobre ESTABLECIMIENTOS (IDs 1-8)")
    print(f"{'='*62}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Servidor detenido.")
        server.shutdown()

if __name__=="__main__":
    main()