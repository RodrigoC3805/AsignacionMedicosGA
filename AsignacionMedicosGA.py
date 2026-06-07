from __future__ import annotations

import json
import random
import statistics
import threading
import urllib.parse
import webbrowser
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
# INSTANCIA FIJA DEL DOCUMENTO — Tabla 2 (sin columna de región)
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
    """
    Genera la instancia del informe con semilla 42.
    Cada médico tiene 3 establecimientos preferidos elegidos al azar entre IDs 1-8.
    """
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

        # establishment_id -> índice 0-based en la lista
        self.est_id_to_idx: Dict[int, int] = {e.id: i for i, e in enumerate(establishments)}

        # Para cada médico: lista de índices 0-based de sus establecimientos preferidos
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
        """
        Matriz satisfacción N x M.
        s[médico][est_idx] = 30 (1ª pref), 20 (2ª), 10 (3ª), 0 (no preferido).
        Preferencias son sobre establecimientos, no regiones (§3.1).
        """
        m = [[0]*self.M for _ in range(self.N)]
        for di, d in enumerate(self.doctors):
            for rank, eid in enumerate(d.preferences):
                m[di][self.est_id_to_idx[eid]] = [30, 20, 10][rank]
        return m

    def rg(self):
        return self.rng.randrange(self.M)

    def init_pop(self):
        """§6: 20% semi-heurístico (1ª pref establecimiento), 80% aleatorio."""
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
            # Mejora 5: guía la mutación hacia un establecimiento preferido del médico
            if self.config.guided_mutation and self.rng.random() < self.config.guided_mutation_prob:
                ch[i] = self.rng.choice(self.pref_indices[i])
            else:
                ch[i] = self.rg()

    def repair(self, ch):
        """Mejora 6: reubica excedentes en centros con vacantes (pref. establecimientos preferidos)."""
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
        """Decodifica cromosoma. Muestra establecimientos preferidos (no regiones)."""
        out = []
        for di, ei in enumerate(ch):
            d, e = self.doctors[di], self.establishments[ei]
            pref_names = [self.establishments[self.est_id_to_idx[eid]].name for eid in d.preferences]
            out.append({
                "doctor_id":                     d.id,
                "doctor_name":                   d.name,
                "preferred_establishments":      d.preferences,      # IDs [3, 1, 7]
                "preferred_establishment_names": pref_names,         # nombres legibles
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
        self.free_ga: Optional[MedicalAssignmentGA] = None
        self.free_optimizing = False
        self.free_breakdown: Optional[FitnessBreakdown] = None
        self.free_decoded: List[Dict] = []
        self.free_config_label = ""
        self.corridas: Dict[int, CorridaState] = {
            n: CorridaState(n, CORRIDA_CONFIGS[n]["label"]) for n in range(1, 5)
        }
        self.corridas_running = False
        self.status_message = "Instancia Tabla 2 cargada (N=30, M=8, B=S/50 000). Lista."
        self.status_type    = "success"

    def _make_config(self, crossover_type="single_point",
                     guided_mutation=False, repair_enabled=False) -> GAConfig:
        return GAConfig(**FIXED_GA_PARAMS, seed=DOCUMENT_SEED,
                        crossover_type=crossover_type,
                        guided_mutation=guided_mutation,
                        repair_enabled=repair_enabled)

    def run_free(self, corrida_num: int):
        if self.free_optimizing: return
        cc  = CORRIDA_CONFIGS[corrida_num]
        cfg = self._make_config(cc["crossover_type"], cc["guided_mutation"], cc["repair_enabled"])
        with self.lock:
            self.free_optimizing   = True
            self.free_breakdown    = None
            self.free_decoded      = []
            self.free_config_label = cc["label"]
            docs, ests, bud        = generate_document_instance()
            self.free_ga           = MedicalAssignmentGA(docs, ests, bud, cfg)
            self.status_message    = f"Ejecutando corrida: {cc['label']}…"
            self.status_type       = "running"

        def job():
            try:
                ch, bd = self.free_ga.evolve()
                with self.lock:
                    self.free_breakdown = bd
                    self.free_decoded   = self.free_ga.decode(ch)
                    self.status_message = (f"[{cc['label']}] Listo en "
                                           f"{self.free_ga.current_gen} gen. "
                                           f"Fitness: {bd.fitness:.0f}")
                    self.status_type    = "success"
            except Exception as e:
                with self.lock:
                    self.status_message = f"Error: {e}"
                    self.status_type    = "error"
            finally:
                with self.lock:
                    self.free_optimizing = False

        threading.Thread(target=job, daemon=True).start()

    def run_all_corridas(self):
        if self.corridas_running: return
        with self.lock:
            self.corridas_running = True
            for n in range(1, 5):
                self.corridas[n] = CorridaState(n, CORRIDA_CONFIGS[n]["label"], status="running")
            self.status_message = "Ejecutando 4 corridas en paralelo…"
            self.status_type    = "running"

        threads = []
        for n in range(1, 5):
            cc  = CORRIDA_CONFIGS[n]
            cfg = self._make_config(cc["crossover_type"], cc["guided_mutation"], cc["repair_enabled"])
            docs, ests, bud = generate_document_instance()
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
                            self.status_message   = "Las 4 corridas finalizaron."
                            self.status_type      = "success"

            t = threading.Thread(target=job, daemon=True)
            threads.append(t)
        for t in threads: t.start()

    def reset_corridas(self):
        with self.lock:
            self.corridas = {n: CorridaState(n, CORRIDA_CONFIGS[n]["label"]) for n in range(1, 5)}
            self.corridas_running = False

    def get_state(self):
        with self.lock:
            free_gp = {}
            if self.free_ga:
                free_gp = {"current_gen": self.free_ga.current_gen,
                            "history_best": self.free_ga.history_best[-300:],
                            "history_mean": self.free_ga.history_mean[-300:]}
            free_bd = None
            if self.free_breakdown:
                b = self.free_breakdown
                free_bd = {"fitness": b.fitness, "satisfaction_total": b.satisfaction_total,
                           "penalty_capacity": b.penalty_capacity,
                           "penalty_epidemiological": b.penalty_epidemiological,
                           "penalty_budget": b.penalty_budget, "total_cost": b.total_cost,
                           "budget_excess": b.budget_excess,
                           "assigned_per_establishment": {str(k): v for k, v in b.assigned_per_establishment.items()}}
            return {
                "n_doctors":         len(self.doctors),
                "n_establishments":  len(self.establishments),
                "budget":            self.budget,
                "seed":              DOCUMENT_SEED,
                "free_optimizing":   self.free_optimizing,
                "free_config_label": self.free_config_label,
                "corridas_running":  self.corridas_running,
                "status_message":    self.status_message,
                "status_type":       self.status_type,
                "establishments":    [asdict(e) for e in self.establishments],
                "free_breakdown":    free_bd,
                "free_decoded":      self.free_decoded,
                "free_ga_progress":  free_gp,
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
.body{display:grid;grid-template-columns:260px 1fr;height:calc(100vh - 88px);overflow:hidden}
.sidebar{overflow-y:auto;padding:14px 12px;border-right:1px solid var(--brd);background:var(--surf);display:flex;flex-direction:column;gap:10px}
.content{overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.btn{display:inline-flex;align-items:center;gap:5px;padding:6px 14px;border-radius:6px;border:none;cursor:pointer;font-family:var(--font);font-size:12px;font-weight:600;transition:all .14s;white-space:nowrap}
.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-ok{background:linear-gradient(135deg,#0fa870,#1fd8a4);color:#fff}
.btn-ok:hover:not(:disabled){filter:brightness(1.1);transform:translateY(-1px)}
.btn-ghost{background:var(--surf3);color:var(--txt2);border:1px solid var(--brd2)}
.btn-ghost:hover:not(:disabled){color:var(--txt);border-color:var(--acc)}
.sec-lbl{font-size:9px;font-weight:700;color:var(--txt3);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;padding-left:2px}
.mg{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.met{background:var(--surf3);border:1px solid var(--brd);border-radius:8px;padding:8px 10px}
.met.full{grid-column:1/-1}
.ml{font-size:9px;color:var(--txt3);font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
.mv{font-size:18px;font-weight:700;font-family:var(--mono);letter-spacing:-.5px}
.ca{color:var(--acc)}.cc{color:var(--acc2)}.co{color:var(--ok)}.cw{color:var(--warn)}.ce{color:var(--err)}.cp{color:var(--acc3)}
.params-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.param-box{background:var(--surf3);border:1px solid var(--brd);border-radius:7px;padding:7px 9px}
.param-box.full{grid-column:1/-1}
.pl{font-size:9px;color:var(--txt3);font-weight:600;text-transform:uppercase;letter-spacing:.4px;margin-bottom:2px}
.pv{font-size:13px;font-weight:700;font-family:var(--mono);color:var(--acc2)}
.param-note{font-size:9px;color:var(--txt3);margin-top:1px}
.corrida-btns{display:flex;flex-direction:column;gap:5px}
.cbtn{display:flex;align-items:center;gap:8px;padding:8px 11px;background:var(--surf3);border:1px solid var(--brd);border-radius:8px;cursor:pointer;font-family:var(--font);font-size:11px;font-weight:500;color:var(--txt2);transition:all .14s;text-align:left}
.cbtn:hover:not(:disabled){border-color:var(--acc);color:var(--txt)}
.cbtn:disabled{opacity:.4;cursor:not-allowed}
.cbtn .num{width:20px;height:20px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0}
.cn1{background:rgba(59,127,255,.2);color:var(--c1)}.cn2{background:rgba(245,166,35,.2);color:var(--c2)}
.cn3{background:rgba(31,216,164,.2);color:var(--c3)}.cn4{background:rgba(176,126,248,.2);color:var(--c4)}
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
.mejoras-table{width:100%;border-collapse:collapse;font-size:11px}
.mejoras-table th{background:var(--surf3);padding:7px 10px;text-align:left;font-size:9px;font-weight:700;color:var(--txt3);text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--brd2)}
.mejoras-table td{padding:8px 10px;border-bottom:1px solid var(--brd);vertical-align:top;line-height:1.5}
.mejoras-table tr:last-child td{border-bottom:none}
.m-num{font-family:var(--mono);font-weight:700;color:var(--acc);width:30px}
.m-name{font-weight:600;color:var(--txt)}
.m-benefit{color:var(--txt2)}
.overlay{position:fixed;inset:0;background:rgba(6,10,18,.85);backdrop-filter:blur(8px);z-index:200;display:flex;align-items:center;justify-content:center;opacity:0;pointer-events:none;transition:opacity .22s}
.overlay.on{opacity:1;pointer-events:all}
.ov-box{background:var(--surf2);border:1px solid var(--brd2);border-radius:14px;padding:28px 36px;text-align:center;max-width:380px}
.spin{width:44px;height:44px;border:3px solid var(--brd2);border-top-color:var(--acc);border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 14px}
@keyframes spin{to{transform:rotate(360deg)}}
.ov-gen{font-family:var(--mono);font-size:11px;color:var(--acc2);margin-top:7px}
.ov-prog{background:var(--brd);border-radius:2px;height:2px;margin-top:8px;overflow:hidden}
.ov-bar{height:100%;background:linear-gradient(90deg,var(--acc),var(--acc2));border-radius:2px;transition:width .5s}
/* Pref tags: distinguen 1ª/2ª/3ª preferencia de ESTABLECIMIENTO */
.pt{font-size:9px;padding:2px 6px;border-radius:3px;background:var(--surf);border:1px solid var(--brd);color:var(--txt3);white-space:nowrap;display:inline-block}
.pt.p1{border-color:var(--ok);color:var(--ok)}
.pt.p2{border-color:var(--warn);color:var(--warn)}
.pt.p3{border-color:var(--err);color:var(--err)}
.fact-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.fact-row{background:var(--surf3);border:1px solid var(--brd);border-radius:7px;padding:7px 9px;display:flex;align-items:center;gap:7px}
.fact-text{font-size:10px;line-height:1.5}
.fact-lbl{color:var(--txt3);font-size:9px}
.fact-val{font-family:var(--mono);font-weight:600;color:var(--ok)}
.info-note{background:rgba(59,127,255,.07);border:1px solid rgba(59,127,255,.25);border-left:3px solid var(--acc);border-radius:8px;padding:10px 14px;font-size:11px;color:var(--txt2);line-height:1.7;margin-bottom:12px}
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
<aside class="sidebar">
  <div class="sec-lbl">Instancia (Tabla 2)</div>
  <div class="mg">
    <div class="met"><div class="ml">Médicos</div><div class="mv ca">30</div></div>
    <div class="met"><div class="ml">Centros</div><div class="mv cc">8</div></div>
    <div class="met full"><div class="ml">Presupuesto MINSA</div><div class="mv cw" style="font-size:15px">S/ 50 000</div></div>
    <div class="met"><div class="ml">Cap. total</div><div class="mv co">38</div><div style="font-size:9px;color:var(--txt3);margin-top:1px">plazas</div></div>
    <div class="met"><div class="ml">Cuota mín.</div><div class="mv ce">19</div><div style="font-size:9px;color:var(--txt3);margin-top:1px">obligatorias</div></div>
    <div class="met full"><div class="ml">K<sub>mín</sub> factible</div><div class="mv" style="font-size:14px;color:var(--ok)">S/ 48 000</div><div style="font-size:9px;color:var(--txt3);margin-top:1px">margen S/ 2 000 (4 %)</div></div>
  </div>

  <div class="sec-lbl" style="margin-top:4px">Parámetros del AG (fijos)</div>
  <div class="params-grid">
    <div class="param-box"><div class="pl">Población</div><div class="pv">100</div></div>
    <div class="param-box"><div class="pl">Max gen.</div><div class="pv">500</div></div>
    <div class="param-box"><div class="pl">P. cruce</div><div class="pv">0.80</div></div>
    <div class="param-box"><div class="pl">P. mutación</div><div class="pv">0.05</div></div>
    <div class="param-box"><div class="pl">Élite</div><div class="pv">2</div><div class="param-note">§5</div></div>
    <div class="param-box"><div class="pl">Torneo k</div><div class="pv">3</div><div class="param-note">§5</div></div>
    <div class="param-box"><div class="pl">Estagnación</div><div class="pv">50</div></div>
    <div class="param-box"><div class="pl">Semi-heurís.</div><div class="pv">20 %</div><div class="param-note">§6</div></div>
    <div class="param-box full"><div class="pl">Mut. guiada prob.</div><div class="pv">0.50</div><div class="param-note">§4.2 Mejora 5</div></div>
    <div class="param-box full"><div class="pl">Reparación prob.</div><div class="pv">0.80</div><div class="param-note">§4.3 Mejora 6</div></div>
    <div class="param-box full"><div class="pl">Semilla</div><div class="pv">42</div><div class="param-note">reproducible §2.1</div></div>
  </div>

  <div class="sec-lbl" style="margin-top:4px">Resultado (AG libre)</div>
  <div class="mg">
    <div class="met full"><div class="ml">Fitness</div><div class="mv ca" id="r-fit">—</div></div>
    <div class="met"><div class="ml">Satisfacción</div><div class="mv co" id="r-sat">—</div></div>
    <div class="met"><div class="ml">Costo bonos</div><div class="mv cw" id="r-cost" style="font-size:12px">—</div></div>
    <div class="met full"><div class="ml">Corrida actual</div><div class="mv cp" id="r-label" style="font-size:12px">—</div></div>
  </div>

  <div class="sec-lbl" style="margin-top:4px">Ejecutar corrida libre</div>
  <div class="corrida-btns">
    <button class="cbtn" id="cbtn-1" onclick="runFree(1)"><span class="num cn1">C1</span><div><div style="font-size:11px;font-weight:600">Línea base</div><div style="font-size:9px;color:var(--txt3)">Cruce 1 punto · sin mejoras extra</div></div></button>
    <button class="cbtn" id="cbtn-2" onclick="runFree(2)"><span class="num cn2">C2</span><div><div style="font-size:11px;font-weight:600">Cruce uniforme</div><div style="font-size:9px;color:var(--txt3)">Mejora 4 activa</div></div></button>
    <button class="cbtn" id="cbtn-3" onclick="runFree(3)"><span class="num cn3">C3</span><div><div style="font-size:11px;font-weight:600">Mutación guiada</div><div style="font-size:9px;color:var(--txt3)">Mejora 5 — pref. establecimiento</div></div></button>
    <button class="cbtn" id="cbtn-4" onclick="runFree(4)"><span class="num cn4">C4</span><div><div style="font-size:11px;font-weight:600">Con reparación</div><div style="font-size:9px;color:var(--txt3)">Mejora 6 activa</div></div></button>
  </div>
</aside>

<main class="content">
  <div class="card" style="flex-shrink:0">
    <div class="card-body" style="padding:6px 10px">
      <div class="tabs">
        <button class="tab active" onclick="switchTab('libre')">📈 AG Libre</button>
        <button class="tab" onclick="switchTab('corridas')">🧪 4 Corridas</button>
        <button class="tab" onclick="switchTab('establecimientos')">🏨 Establecimientos</button>
        <button class="tab" onclick="switchTab('asignaciones')">👨‍⚕️ Asignaciones</button>
        <button class="tab" onclick="switchTab('desglose')">📊 Desglose</button>
        <button class="tab" onclick="switchTab('informe')">📄 Informe</button>
      </div>
    </div>
  </div>

  <!-- AG LIBRE -->
  <div class="tc active card" id="tc-libre">
    <div class="card-hdr">
      <div class="card-title">📈 Convergencia — AG Libre</div>
      <span style="font-size:10px;color:var(--txt3);font-family:var(--mono)" id="gen-ctr">Gen: —</span>
    </div>
    <div class="card-body">
      <div class="ch"><canvas id="cvs-libre"></canvas></div>
      <div class="chart-legend">
        <span><span class="leg-line" style="background:var(--acc)"></span>Mejor fitness</span>
        <span><span class="leg-line" style="background:var(--acc2);opacity:.6"></span>Fitness promedio</span>
      </div>
      <div class="empty" id="cvs-libre-empty" style="margin-top:12px">
        <div class="ic">📉</div><h3>Sin datos</h3>
        <p>Selecciona una corrida en el panel lateral para ejecutar el AG.</p>
      </div>
    </div>
  </div>

  <!-- 4 CORRIDAS -->
  <div class="tc card" id="tc-corridas">
    <div class="card-hdr">
      <div class="card-title">🧪 4 Corridas Experimentales en Paralelo</div>
      <div style="display:flex;gap:7px;align-items:center">
        <button class="btn btn-ok" id="btn-corridas" onclick="runCorridas()">▶ Ejecutar las 4 juntas</button>
        <button class="btn btn-ghost" onclick="resetCorridas()">🗑 Reiniciar</button>
      </div>
    </div>
    <div class="card-body">
      <p style="font-size:11px;color:var(--txt2);margin-bottom:12px;line-height:1.6">
        Las 4 corridas usan la <b>instancia exacta de la Tabla 2</b> y la <b>misma semilla (42)</b>, variando un único factor por corrida (Tabla 3 del informe). Corren en hilos independientes en paralelo.
      </p>
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
          <th>Corrida</th><th>Factor</th><th>Hipótesis §8.1</th><th>Estado</th>
          <th>Fitness</th><th>Satisf.</th><th>P.Cap</th><th>P.Epi</th><th>P.Ppto</th><th>Costo</th><th>Gen.</th>
        </tr></thead>
        <tbody id="cr-tbody"></tbody></table>
      </div>
      <div class="corridas-grid" id="corridas-grid"></div>
    </div>
  </div>

  <!-- ESTABLECIMIENTOS -->
  <div class="tc card" id="tc-establecimientos">
    <div class="card-hdr"><div class="card-title">🏨 Establecimientos — Tabla 2 del informe</div></div>
    <div class="card-body">
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
      <div class="card-title">👨‍⚕️ Asignación Individual — AG Libre</div>
      <span id="a-cnt" style="font-size:10px;color:var(--txt3)">0 asig.</span>
    </div>
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
      <div class="empty"><div class="ic">🩺</div><h3>Sin asignaciones</h3><p>Ejecuta una corrida en el panel lateral.</p></div>
    </div>
  </div>

  <!-- DESGLOSE -->
  <div class="tc card" id="tc-desglose">
    <div class="card-hdr"><div class="card-title">📊 Desglose del Fitness — AG Libre</div></div>
    <div class="card-body">
      <div class="empty" id="bd-empty"><div class="ic">📐</div><h3>Sin resultados</h3><p>Ejecuta una corrida.</p></div>
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
          <span style="color:var(--err)">P<sub>cap</sub> = 1000 · Σ max(0, n_j − C_j)           [Mejora 2 — proporcional]</span><br>
          <span style="color:var(--err)">P<sub>epi</sub> = Σ w_j · max(0, Q_j − n_j)  w={Alto:500, Medio:250, Bajo:100}  [Mejora 3]</span><br>
          <span style="color:var(--warn)">P<sub>pre</sub> = 0.5 · max(0, K(x) − B)              [Mejora 2 — lineal]</span>
        </div>
      </div>
    </div>
  </div>

  <!-- INFORME -->
  <div class="tc card" id="tc-informe">
    <div class="card-hdr"><div class="card-title">📄 Resumen del Informe — Grupo 6 UNMSM 2026-I</div></div>
    <div class="card-body">
      <div class="info-note">
        <b style="color:var(--acc)">§2.1 — Modelo de preferencias (sobre establecimientos)</b><br>
        Cada médico posee un ranking de <b>3 establecimientos preferidos</b> (1ª, 2ª y 3ª opción), elegidos entre los M = 8 centros de la Tabla 2 mediante semilla 42. La función de aptitud §3.1 puntúa: <b>+30</b> pts si se asigna al médico su 1ª preferencia, <b>+20</b> pts la 2ª, <b>+10</b> pts la 3ª, y <b>0</b> pts si el establecimiento asignado no figura entre sus tres opciones.
      </div>
      <div style="margin-bottom:14px">
        <div style="font-size:10px;font-weight:700;color:var(--txt3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px">6 Mejoras propuestas (§9 · Tabla 4)</div>
        <table class="mejoras-table" style="border:1px solid var(--brd);border-radius:8px;overflow:hidden">
          <thead><tr><th>N°</th><th>Mejora</th><th>Beneficio principal</th><th>Sección</th></tr></thead>
          <tbody>
            <tr><td class="m-num">M1</td><td class="m-name">Verificación previa de factibilidad</td><td class="m-benefit">Evita correr sobre instancias imposibles</td><td class="tm" style="color:var(--txt3)">§2.2</td></tr>
            <tr><td class="m-num">M2</td><td class="m-name">Penalizaciones proporcionales</td><td class="m-benefit">Crea gradiente; elimina mesetas de fitness</td><td class="tm" style="color:var(--txt3)">§3.2–3.4</td></tr>
            <tr><td class="m-num">M3</td><td class="m-name">Cuota mínima ponderada por riesgo</td><td class="m-benefit">Cobertura más equilibrada y realista</td><td class="tm" style="color:var(--txt3)">§3.3</td></tr>
            <tr><td class="m-num">M4</td><td class="m-name">Cruce uniforme configurable</td><td class="m-benefit">Mejor mezcla en codificación de asignación</td><td class="tm" style="color:var(--txt3)">§4.1</td></tr>
            <tr><td class="m-num">M5</td><td class="m-name">Mutación guiada por preferencia de establecimiento</td><td class="m-benefit">Dirige exploración hacia alta satisfacción</td><td class="tm" style="color:var(--txt3)">§4.2</td></tr>
            <tr><td class="m-num">M6</td><td class="m-name">Operador de reparación de sobrecapacidad</td><td class="m-benefit">Acelera convergencia a soluciones factibles</td><td class="tm" style="color:var(--txt3)">§4.3</td></tr>
          </tbody>
        </table>
      </div>
      <div style="margin-bottom:14px">
        <div style="font-size:10px;font-weight:700;color:var(--txt3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px">Plan de corridas (§8.1 · Tabla 3)</div>
        <table class="mejoras-table" style="border:1px solid var(--brd);border-radius:8px;overflow:hidden">
          <thead><tr><th>Corrida</th><th>Configuración</th><th>Factor estudiado</th><th>Hipótesis</th></tr></thead>
          <tbody>
            <tr><td class="m-num" style="color:var(--c1)">C1</td><td class="m-name">Parámetros base</td><td class="m-benefit">Referencia de comparación</td><td style="color:var(--txt3);font-size:10px">—</td></tr>
            <tr><td class="m-num" style="color:var(--c2)">C2</td><td class="m-name">Cruce uniforme vs. un punto</td><td class="m-benefit">Tipo de cruzamiento (M4)</td><td style="color:var(--txt3);font-size:10px">Mejor mezcla genética</td></tr>
            <tr><td class="m-num" style="color:var(--c3)">C3</td><td class="m-name">Mutación guiada por establecimiento preferido</td><td class="m-benefit">Mutación dirigida (M5)</td><td style="color:var(--txt3);font-size:10px">Mayor satisfacción</td></tr>
            <tr><td class="m-num" style="color:var(--c4)">C4</td><td class="m-name">Con operador de reparación</td><td class="m-benefit">Manejo de restricciones (M6)</td><td style="color:var(--txt3);font-size:10px">Converge más rápido</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

</main>
</div>
</div>

<div class="overlay" id="overlay">
  <div class="ov-box">
    <div class="spin"></div>
    <div style="font-size:16px;font-weight:700;margin-bottom:4px">Optimizando…</div>
    <div style="font-size:11px;color:var(--txt2)" id="ov-label">El AG evalúa poblaciones, cruces y mutaciones.</div>
    <div class="ov-gen" id="ov-gen">Generación: 0</div>
    <div class="ov-prog"><div class="ov-bar" id="ov-bar" style="width:0%"></div></div>
  </div>
</div>

<script>
let S={},allSol=[],pollInt=null;
let charts={libre:null,compare:null};
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
  const order=['libre','corridas','establecimientos','asignaciones','desglose','informe'];
  document.querySelectorAll('.tab')[order.indexOf(t)].classList.add('active');
  document.getElementById('tc-'+t).classList.add('active');
  if(t==='libre')setTimeout(()=>charts.libre&&charts.libre.resize(),10);
  if(t==='corridas')setTimeout(()=>charts.compare&&charts.compare.resize(),10);
}
function fm(v){return'S/ '+Number(v).toLocaleString('es-PE');}
function rb(r){const m={'Alto':'ba','Medio':'bm','Bajo':'bb'},d={'Alto':'🔴','Medio':'🟡','Bajo':'🟢'};return`<span class="bdg ${m[r]||''}">${d[r]||''} ${r}</span>`;}
function pb(p){const m={30:'b30',20:'b20',10:'b10',0:'b0'},l={30:'30·1ª',20:'20·2ª',10:'10·3ª',0:'0·—'};return`<span class="bdg ${m[p]||'b0'}">${l[p]||'—'}</span>`;}

function renderFreeResult(s){
  if(!s.free_breakdown){['r-fit','r-sat','r-cost'].forEach(id=>document.getElementById(id).textContent='—');document.getElementById('r-label').textContent='—';return;}
  const b=s.free_breakdown;
  document.getElementById('r-fit').textContent=b.fitness.toFixed(0);
  document.getElementById('r-sat').textContent=b.satisfaction_total;
  document.getElementById('r-cost').textContent=fm(b.total_cost);
  document.getElementById('r-label').textContent=s.free_config_label||'—';
}
function renderConvergence(s){
  const em=document.getElementById('cvs-libre-empty'),cv=document.getElementById('cvs-libre');
  const gp=s.free_ga_progress;
  if(!gp||!gp.history_best||gp.history_best.length<2){em.style.display='block';cv.style.display='none';document.getElementById('gen-ctr').textContent='Gen: —';return;}
  em.style.display='none';cv.style.display='block';
  document.getElementById('gen-ctr').textContent=`Gen: ${gp.current_gen}`;
  if(charts.libre){charts.libre.data={best:gp.history_best,mean:gp.history_mean};drawSingle(charts.libre);}
}
function renderEst(s){
  const asgn=s.free_breakdown?s.free_breakdown.assigned_per_establishment:{};
  document.getElementById('est-grid').innerHTML=s.establishments.map((e,i)=>{
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
  document.getElementById('a-cnt').textContent=`${allSol.length} asig.`;
  applyF();
}
function applyF(){
  const nq=document.getElementById('flt-n').value.toLowerCase(),rq=document.getElementById('flt-r').value,pq=document.getElementById('flt-p').value;
  const f=allSol.filter(x=>(!nq||(x.doctor_name+x.assigned_establishment_name).toLowerCase().includes(nq))&&(!rq||x.assigned_risk_level===rq)&&(!pq||String(x.satisfaction_points)===pq));
  document.getElementById('flt-cnt').textContent=f.length?`${f.length} resultados`:'';
  const w=document.getElementById('tbl-wrap');
  if(!allSol.length){w.innerHTML='<div class="empty"><div class="ic">🩺</div><h3>Sin asignaciones</h3><p>Ejecuta una corrida desde el panel lateral.</p></div>';return;}
  if(!f.length){w.innerHTML='<div class="empty"><div class="ic">🔍</div><h3>Sin resultados</h3><p>Ajusta los filtros.</p></div>';return;}
  // Columnas de preferencias muestran NOMBRES de establecimientos
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
  document.getElementById('cr-tbody').innerHTML=[1,2,3,4].map(n=>{
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
  if(Object.keys(cmpData).length)drawCompare(cmpData);
  const confTxt={
    1:'Cruce: un punto · Mutación: aleatoria · Sin reparación',
    2:`<b style="color:var(--c2)">Cruce: uniforme (M4)</b> · Mutación: aleatoria · Sin reparación`,
    3:`Cruce: un punto · <b style="color:var(--c3)">Mutación: guiada por estab. preferido (M5)</b> · Sin reparación`,
    4:`Cruce: un punto · Mutación: aleatoria · <b style="color:var(--c4)">Reparación activa (M6)</b>`,
  };
  document.getElementById('corridas-grid').innerHTML=[1,2,3,4].map(n=>{
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
    return`<div class="corrida-card">
      <div class="corrida-hdr"><div style="display:flex;align-items:center"><div class="num cn${n}">${n}</div><div><div class="corrida-label">${c.label}</div><div class="corrida-factor">${FACTORS[n]}</div></div></div><span class="cs ${stCls}">${stTxt}</span></div>
      <div class="corrida-body"><div style="font-size:9px;color:var(--txt3);margin-bottom:8px;line-height:1.8">${confTxt[n]}</div>${mH}${mini}</div>
    </div>`;
  }).join('');
  [1,2,3,4].forEach(n=>{
    const c=crs[String(n)];
    if(c.ga_progress?.history_best?.length>=2){const mc=mkChart(`mini-${n}`);if(mc){mc.data={best:c.ga_progress.history_best,mean:[]};drawSingle(mc,COLORS[n]);}}
  });
}
function renderAll(s){
  S=s;
  renderFreeResult(s);renderConvergence(s);renderEst(s);renderAssign(s);renderDesglose(s);renderCorridas(s);
  document.getElementById('smsg').textContent=s.status_message;
  document.getElementById('sdot').className='sdot '+(s.status_type||'info');
  const busy=s.free_optimizing||s.corridas_running;
  document.getElementById('overlay').classList.toggle('on',s.free_optimizing&&!s.corridas_running);
  if(s.free_optimizing&&s.free_ga_progress){
    const gen=s.free_ga_progress.current_gen||0;
    document.getElementById('ov-gen').textContent=`Generación: ${gen}`;
    document.getElementById('ov-label').textContent=`Corrida: ${s.free_config_label||'…'}`;
    document.getElementById('ov-bar').style.width=`${Math.min(100,(gen/500)*100).toFixed(1)}%`;
  }
  [1,2,3,4].forEach(n=>{const b=document.getElementById(`cbtn-${n}`);if(b)b.disabled=busy;});
  const bc=document.getElementById('btn-corridas');if(bc)bc.disabled=s.corridas_running;
}
async function fetchState(){try{renderAll(await(await fetch('/api/state')).json());}catch(e){}}
async function runFree(n){await fetch(`/api/run_free?corrida=${n}`,{method:'POST'});switchTab('libre');startPoll();}
async function runCorridas(){await fetch('/api/run_corridas',{method:'POST'});switchTab('corridas');startPoll();}
async function resetCorridas(){await fetch('/api/reset_corridas',{method:'POST'});await fetchState();}
function startPoll(){
  if(pollInt)clearInterval(pollInt);
  pollInt=setInterval(async()=>{await fetchState();if(!S.free_optimizing&&!S.corridas_running){clearInterval(pollInt);pollInt=null;}},700);
}
window.addEventListener('load',async()=>{
  charts.libre=mkChart('cvs-libre');charts.compare=mkChart('cvs-compare');
  await fetchState();startPoll();
});
window.addEventListener('resize',()=>{charts.libre&&charts.libre.resize();charts.compare&&charts.compare.resize();});
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
        if parsed.path=="/api/run_free":
            STATE.run_free(int(params.get("corrida",["1"])[0]));self.send_json({"ok":True})
        elif parsed.path=="/api/run_corridas":
            STATE.run_all_corridas();self.send_json({"ok":True})
        elif parsed.path=="/api/reset_corridas":
            STATE.reset_corridas();self.send_json({"ok":True})
        else:self.send_response(404);self.end_headers()


def main():
    port=8765
    server=HTTPServer(("localhost",port),Handler)
    url=f"http://localhost:{port}"
    print(f"\n{'='*62}")
    print(f"  AG SERUMS — Asignación de Médicos Rurales · MINSA")
    print(f"  Grupo 6 · Sistemas Inteligentes 2026-I · UNMSM")
    print(f"{'='*62}")
    print(f"  URL : {url}")
    print(f"  Instancia : Tabla 2 (N=30, M=8, B=S/50 000)")
    print(f"  Preferencias: sobre ESTABLECIMIENTOS (IDs 1-8) — §2.1 §3.1")
    print(f"  Ctrl+C para detener.")
    print(f"{'='*62}\n")
    threading.Timer(0.8,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:print("\n  Servidor detenido.");server.shutdown()

if __name__=="__main__":
    main()