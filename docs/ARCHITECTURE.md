# Courtroom Jargon Game — Architecture & Pipelines

---

## 1. Le jeu en une page

Le joueur se réveille dans une **salle d'audience**, accusé d'une faute liée à son métier. Trois personnages — **juge, procureur, avocat** — débattent de son cas en utilisant un **vocabulaire technique précis** qui est un jargon choisit par le joueur en début de partie et qui n'est pas lié à l'affaire discuté dans le tribunal (YouTubeur, influenceur, médecin, aviation, politique, cinéma…), mêlé au registre judiciaire.

- Le joueur **choisit le thème** en début de partie.
- À la fin, il doit **deviner, en langage courant, son métier et sa faute**.
- La partie est **courte** (budget de tours limité).
- **100 % local, hors-ligne**, sur la machine du joueur (via `llama.cpp`).

Langue du jeu : **anglais**.

> ### 📌 Statut d'implémentation (v1) — écarts verrouillés
> Décisions prises lors du refactor (priment sur le texte historique ci-dessous) :
> 1. **Jargon = écran de fumée.** Le §1 fait foi : le jargon choisi **n'est pas lié**
>    au métier/à la faute cachés. L'adaptateur LoRA est donc indexé par **style de
>    jargon** (corporate, aviation…), **pas** par métier. → corrige §4.2 (« le métier
>    choisi ») et l'exemple §6 (profession **indépendante** du jargon).
> 2. **Pas de loquet déterministe en v1.** L'anti-spoiler regex, le calendrier et le
>    registre d'indices (§4.3) sont **abandonnés** : on fait confiance au GM. Seul levier
>    de clôture = **pression de tours injectée dans le prompt** + le flag `wrap_up` du GM.
>    Les `clues` restent du matériau que le GM *peut* utiliser, jamais forcé par le code.
> 3. **Entraînement sur Modal GPU.** Scripts dans `training/` (teacher_datagen, finetune,
>    convert_gguf). `pip install modal` puis `modal token new`.
> 4. **TTS conservé en option** (hors chemin critique, fallback texte).

> ### 📌 Statut final (v2) — architecture livrée (prime sur tout le texte historique ci-dessous)
> 1. **Un seul modèle 1B pour tout le jeu.** Le Game Master n'est plus un 4B séparé : c'est
>    **MiniCPM5-1B + un LoRA « director »** (multitâche : Case File + décisions de tour + scoring,
>    sous grammaire GBNF). Les acteurs = le même base + un LoRA **par style**. Tout le jeu =
>    un base ~1B + deux adaptateurs ~90 Mo, échangés par référence.
> 2. **Pourquoi le 4B a été abandonné :** ~1 tok/s sur un CPU 2 vCPU (≈4 min/décision) — non
>    viable. Le rôle du directeur a été **distillé** dans le 1B (teacher Gemma 4 31B sur Modal,
>    `training/director_*.py`).
> 3. **100 % CPU, sans GPU.** Déploiement en **Docker Space** (`cpu-basic`, 2 vCPU) — Docker sert
>    uniquement à compiler `llama-cpp-python` en **AVX2** (sinon ~2 tok/s). No-think imposé par la
>    grammaire (`root ::= "{"`) ; cache de préfixe KV pour la boucle de tours.
> 4. **Évaluation in-distribution.** Le directeur est benchmarké sur des contextes **held-out**
>    réels du teacher (jamais des stubs ni une seule trajectoire greedy) —
>    `training/director_evaluate.py` / `director_gate.py`. Récit complet et leçons :
>    [`docs/FIELD_NOTES.md`](FIELD_NOTES.md).

---

## 2. Vue d'ensemble

```
                          ┌─────────────────────────────────────────────┐
   seed ───────────────►  │   GAME MASTER  (Nemotron 3 Nano 4B, vanilla) │
                          │   = Scénariste + Directeur + Garde-fous      │
                          │   - écrit le Case File caché (depuis la seed)│
                          │   - dirige le débat dans un budget de tours  │
                          │   - sortie STRUCTURÉE (JSON via grammaire GBNF)
                          └───────┬───────────────────────────▲─────────┘
                                  │ direction de scène          │ transcript + état
                                  ▼                             │
                          ┌─────────────────────────────────────────────┐
                          │   ACTEURS  (MiniCPM5-1B + adaptateur métier)  │
                          │   1 base + 1 LoRA (le métier choisi)          │
                          │   rôle = system prompt (juge/proc./avocat)    │
                          └───────┬───────────────────────────────────────┘
                                  │ réplique en personnage + jargon
                                  ▼
                          ┌─────────────────────────────────────────────┐
                          │   LOQUET DÉTERMINISTE (code)                  │
                          │   - check anti-spoiler dur (regex)            │
                          │   - compteur de tours / registre d'indices    │
                          └───────┬───────────────────────────────────────┘
                                  ▼
                               JOUEUR  ──►  devinette finale ──► scoring
```

**Hors-ligne (jamais chez le joueur) :** `Gemma 4 31B-it` génère les **données d'entraînement** des adaptateurs acteurs. Voir §7.

---

## 3. Les modèles

| Modèle | Rôle | Où | Pourquoi ce modèle |
|---|---|---|---|
| **MiniCPM5-1B** | Les 3 **acteurs** (juge/procureur/avocat) | Runtime, local, `llama.cpp` | Archi **`LlamaForCausalLM`** → LoRA + GGUF sans friction. SOTA de sa classe 1B, GGUF officiel, mode *thinking*, 131K ctx, Apache-2.0. Léger (~0,8 Go en Q4). |
| **Nemotron 3 Nano 4B** | Le **Game Master** (scénariste + directeur + garde-fous) | Runtime, local, `llama.cpp` | Raisonnement fort, **ciblé par NVIDIA pour les « AI gaming NPCs » en edge**, GGUF officiel, tient dans 8 Go, 262K ctx. Hybride Mamba-2 — voir nuance ci-dessous. |
| **Gemma 4 31B-it** | **Teacher** : génération des données synthétiques | **Hors-ligne uniquement** (Modal) | Gros modèle de qualité pour produire des transcriptions riches en jargon. Ne tourne **jamais** chez le joueur. 256K ctx, multilingue. |

### Nuance architecture (importante)
Nemotron 3 Nano 4B est un **hybride Mamba-2** (21 couches Mamba / 4 attention / 17 MLP). Le tooling **LoRA + GGUF** des hybrides est moins mûr → on **ne LoRA-tune jamais Nemotron**. Le Game Master tourne **vanilla** (inférence seule, parfaitement supportée par `llama.cpp`). Toute la charge LoRA repose sur les acteurs **MiniCPM5 (archi Llama)**, faite pour ça.

➡️ **Le découpage par famille de modèle est un choix, pas un accident** : chaque architecture est placée là où elle est bonne.

---

## 4. Les rôles au runtime

### 4.1 Game Master — `Nemotron 3 Nano 4B` (vanilla, sortie GBNF)
Re-mergé en **un seul cerveau** (décision validée). Il assume trois fonctions :

1. **Scénariste** : depuis la `seed`, écrit le **Case File caché** (§6) — la vérité (métier + faute), les faits, le set d'indices, la difficulté. *Une fois, au lancement.*
2. **Directeur** : à **chaque tour**, décide le prochain *beat* (qui parle, quoi, quel indice révéler maintenant) **dans un budget de tours** fixe.
3. **Garde-fous** : il **possède** la régulation (rester sur les rails, pas de spoiler, convergence des indices).

> **Raisonnement du re-merge** : l'agent qui *invente* les indices est le mieux placé pour les *révéler* de façon cohérente. On avait splitté scénariste/directeur quand le directeur était un 1B trop faible ; avec un 4B capable, le split n'a plus de raison d'être. Partie courte ⇒ le coût « 4B à chaque tour » reste borné.

**Sortie structurée obligatoire** (via grammaire **GBNF** de `llama.cpp`) — le GM ne produit pas de prose libre, mais une décision que le moteur peut exécuter :

```json
{
  "next_speaker": "prosecutor",        // judge | prosecutor | lawyer
  "beat_type": "introduce_evidence",   // deck fini : voir §6
  "clue_to_release": "clue_03",        // id d'un indice du Case File, ou null
  "intensity": 3,                      // 1..5
  "stage_direction": "Press on the unexplained spike; cite the metric, stay oblique.",
  "wrap_up": false                     // true quand on doit basculer vers la clôture
}
```

> **Pourquoi GBNF** : un petit modèle *invente* mal mais *classe* bien. En forçant un JSON à champs énumérés, on transforme « diriger » en « choisir dans un deck » → agentique **mais** contrôlable, jamais de dérive chaotique.

### 4.2 Acteurs — `MiniCPM5-1B` + adaptateur de **style de jargon**
> **v1 :** l'adaptateur est indexé par **style de jargon** (écran de fumée), pas par
> métier — le métier caché est indépendant du jargon. Remplacer « métier » par « style »
> partout dans cette section.
- **Une seule base** MiniCPM5-1B chargée, **un seul adaptateur LoRA** par partie : celui du **style de jargon choisi**.
- Les **3 rôles** = **3 system prompts** différents sur la même base + le même adaptateur.
- Entrée d'un acteur à un tour = `system prompt rôle` + `stage_direction` du GM + faits pertinents **formulés en oblique** + contraintes (jargon du thème, anglais, longueur max, **interdiction de nommer le métier/la faute en clair**).

> **Pourquoi 1 LoRA par jargon + 1 system prompt par rôle** : ajouter un métier = entraîner **un** adaptateur ; changer de rôle = **gratuit** (prompt). Et dans une partie, les 3 rôles partagent l'unique adaptateur du métier ⇒ **un seul adaptateur chargé**, idéal pour le local.

### 4.3 Loquet déterministe (code, pas un agent)
> **v1 : section ENTIÈREMENT DIFFÉRÉE.** Pas d'anti-spoiler regex ni de
> calendrier/registre d'indices. On fait confiance au GM ; la seule régulation est la
> **pression de tours injectée dans le prompt** (« tour X / B, converge ») + `wrap_up`.
> À reconsidérer si le GM dérive en pratique.
- **Anti-spoiler dur** : avant d'afficher une réplique, une **regex/string-match** vérifie que la réponse en clair (nom du métier, mots-clés de la faute) **n'apparaît jamais**. Une regex *garantit* ce qu'un modèle ne peut que *promettre*.
- **Compteur de tours + registre d'indices** : état déterministe que le GM **lit**. Si on est en retard sur le calendrier d'indices → on **force** l'indice prévu ; à `budget-1` → on bascule en clôture.

> **Pourquoi un loquet malgré « garde-fous = GM »** : le GM *décide*, mais reste probabiliste. Le loquet est sa **ceinture de sécurité** déterministe, pas un rôle concurrent.

---

## 5. Boucle de jeu (runtime)

```
1. Le joueur choisit un jargon + une difficulté.
2. seed ─► GM (Nemotron) écrit le Case File caché + fixe le budget de tours B (dans une plage fixe).
3. Le moteur charge la base MiniCPM5 + l'adaptateur du métier choisi.
4. Boucle de tours (jusqu'à B) :
   a. GM ─► décision structurée (next_speaker, beat, clue_to_release, intensity...).
   b. Loquet : MAJ registre d'indices/tours ; si retard, force l'indice prévu.
   c. Moteur assemble le prompt de l'acteur (rôle + stage_direction + faits obliques).
   d. Acteur (MiniCPM5 + adaptateur) ─► réplique en personnage + jargon.
   e. Loquet anti-spoiler : accepte / régénère.
   f. Affiche au joueur ; le joueur peut poser une question/contester (réinjecté au GM).
5. À la clôture, le joueur soumet sa devinette (métier + faute, langage courant).
6. Jugement de la devinette : match sémantique contre le Case File caché ─► score.
```

**Squelette de phases** (template **générique déterministe**, pas réécrit par scénario) : `ouverture → charges → défense → échange → clôture`. Le GM place les indices *dans* ces phases selon le budget.

---

## 6. Le Case File caché & l'économie d'indices

> **v1 :** la `profession` est **indépendante** du style de jargon choisi (écran de
> fumée). L'exemple ci-dessous (`airline_pilot` + clue « aviation domain ») suppose à
> tort que jargon = métier — à lire comme : le métier peut être *n'importe quoi*, et les
> indices pointent vers ce métier, pas vers le jargon. L'**économie d'indices** (calendrier
> sur B tours) n'est **pas** ordonnancée par le code en v1 : les `clues` sont du matériau
> optionnel pour le GM (voir §4.3).

Le GM génère, **figé par la seed**, un objet **jamais montré au joueur** :

```json
{
  "profession": "airline_pilot",
  "fault_plain": "skipped a mandatory pre-flight checklist item",
  "facts": ["...", "...", "..."],
  "clues": [
    {"id": "clue_01", "text_oblique": "the crew failed to cross-check the trim setting", "reveals": "aviation domain"},
    {"id": "clue_02", "text_oblique": "a V1 call was made but the rejected-takeoff brief was skipped", "reveals": "the fault"}
  ],
  "severity": 4,
  "difficulty": "normal",
  "turn_budget": 12
}
```

- **`fault_plain` / `profession`** = la **vérité** (sert au loquet anti-spoiler et au scoring final). **Jamais** envoyée aux acteurs en clair.
- **`clues`** = l'**économie d'indices** : `K` indices à étaler sur `B` tours. Le calendrier (quel indice à quelle phase) garantit que le mystère est **soluble** à la fin et **monte en clarté**. C'est ce qui remplace une « planification long terme » : sur une partie courte, **étaler K indices sur B tours + conclure à B** est un simple problème d'ordonnancement géré par le loquet.

**Deck de `beat_type`** (fini, énuméré dans la grammaire GBNF) — exemples :
`opening_statement, file_charge, introduce_evidence, call_witness, objection, escalate, defense_plea, cross_examine, closing, hold_back`.

---

## 7. Pipeline d'entraînement

> C'est le cœur du travail. Sortie finale : **N adaptateurs LoRA GGUF** (un par métier), tous des deltas contre **la même base vanilla MiniCPM5-1B**.

### 7.1 Génération des données synthétiques (Gemma 4 31B-it, hors-ligne sur Modal)
On **distille** : un gros modèle produit des données, le petit modèle les apprend.

Deux jeux de données à produire :
1. **Dataset juridique générique** (partagé) : transcriptions de procès en anglais — procédure, objections, registre judiciaire — **sans jargon métier**. Sert au **stage 1** (§7.3).
2. **Dataset par métier** (×N) : transcriptions de procès **saturées du jargon du métier**, avec des **cas variés** (≠ une seule affaire répétée, sinon l'adaptateur mémorise au lieu de généraliser le jargon). Sert au **stage 2**.

**Garde-fous de génération** (le teacher 31B reste faillible) :
- **Ancrage** : injecter dans le prompt de Gemma le vocabulaire réel du métier (glossaire). *Décision projet : on teste d'abord **sans** glossaire (Gemma 31B est compétent) ; fallback = générer un glossaire avec un modèle puissant et le réinjecter.*
- **Filtrage** : longueur, anglais correct, présence effective des termes attendus, **pas de spoiler de la solution**, format respecté.

Implémenté : servir Gemma 4 31B **en FP8** (`RedHatAI/gemma-4-31B-it-FP8-block`) sur Modal via
`training/teacher_datagen.py` (vLLM, L40S — FP8 natif Ada/Hopper) pour générer en batch.
Prompts **paramétrés et seedés** (métier × faute × ton × disposition × sévérité × nb de tours)
→ données variées et runs reproductibles.

### 7.2 Schéma des données (format ShareGPT / chat template)
Chaque exemple = une **conversation multi-tours** (compatible `apply_chat_template`). Le `system` encode le rôle + le brief caché ; le dataset d'un métier couvre **les 3 rôles**.

```json
{
  "conversations": [
    {"role": "system", "content": "You are the PROSECUTOR in a courtroom. Domain: <profession>. Speak in-character with precise <profession> jargon. Never name the profession or state the fault in plain words. Hidden brief: <oblique facts>."},
    {"role": "user", "content": "<previous turn / judge prompt>"},
    {"role": "assistant", "content": "<the prosecutor's in-character, jargon-rich line>"}
  ]
}
```

> Note : le **comportement de rôle** vient du system prompt, mais doit être **présent dans les exemples** ⇒ chaque adaptateur métier ré-ancre aussi le registre judiciaire (pas besoin d'un LoRA « courtroom » séparé empilé à l'inférence).

### 7.3 Curriculum LoRA en 2 étapes (checkpoint-fork, PAS de merge)

**Stage 1 — adaptateur juridique de base (une seule fois) :**
- Entraîner un LoRA de config **C** (`r`, `alpha`, `target_modules`) sur le **dataset juridique générique**.
- Sauver l'adaptateur → `legal_base` (checkpoint PEFT).

**Stage 2 — fork par métier (×N) :**
- **Initialiser** un LoRA depuis les poids de `legal_base` (config **C identique** — on **ne change pas** le rang en cours de route).
- Lancer un **NOUVEAU run** (optimizer + warmup **frais**) sur le **dataset du métier**.
- Sauver → `actor_<metier>`.

> ⚠️ **Piège à éviter** : ce stage 2 = « charger l'adaptateur comme **initialisation** + nouveau run », **PAS** `trainer.resume_from_checkpoint`. Ce dernier reprend le *même* run avec le *même* scheduler (il sert à la reprise après préemption, pas à un curriculum). Le `check_for_existing_checkpoint` du fichier d'exemple est pour la préemption — à ne pas confondre.

> **Pourquoi checkpoint-fork plutôt que merge dans la base** : on garde la **base MiniCPM5 vanilla**. `llama.cpp` charge alors **1 base + N adaptateurs**, et on peut utiliser le **GGUF officiel** de la base. (Le merge donnerait une base custom à distribuer + un re-entraînement de tous les adaptateurs si on touche au stage juridique.) Coût accepté : chaque adaptateur porte *juridique + jargon* dans son budget de rang — sur un 1B en `r=16–32`, ça tient large.

### 7.4 Hyperparamètres & config LoRA (points de départ)
- **`target_modules`** (Llama) : `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- **`r`** : 16–32 ; **`alpha`** : = `r` (ou 2×`r`) ; **`dropout`** : 0.0.
- **Quantif d'entraînement** : 4-bit (QLoRA) suffit pour un 1B.
- **LR** ~2e-4, scheduler cosine, warmup ~5 %, optim `adamw_8bit`.
- **Identique** entre stage 1 et stage 2 (obligatoire pour le fork).
- **Infra** : Modal + **Unsloth** (TRL `SFTTrainer` + PEFT). Implémenté dans `training/finetune.py` — Unsloth supporte MiniCPM5 (archi Llama), OpenBMB documente officiellement le chemin.

### 7.5 Conversion GGUF (pour `llama.cpp`)
- Convertir chaque adaptateur PEFT (`adapter_config.json` + `adapter_model.safetensors`) en **GGUF mono-fichier** avec `convert_lora_to_gguf.py` (llama.cpp) / outillage *GGUF-my-LoRA*.
- Appliqué **sans merge**, sur **base quantifiée résidente**.
- ✅ **À valider** : qualité d'un adaptateur sur base très quantifiée (Q4) vs Q8.

---

## 8. Service / inférence locale (`llama.cpp`)

Deux modèles résidents (≈ **~3,5 Go en Q4**, tient dans 8 Go) :

**A. Game Master — Nemotron 4B (vanilla)**
- `llama-server` (ou binding), modèle GGUF Nemotron.
- **Décodage contraint par grammaire GBNF** = le schéma JSON de §4.1.
- Pas de LoRA.

**B. Acteurs — MiniCPM5-1B + adaptateur métier**
- `llama-server` avec la base GGUF MiniCPM5 + les adaptateurs chargés :
  - démarrage : `--lora actor_<metier>.gguf` (un suffit par partie), option `--lora-init-without-apply`.
  - par requête : activer via liste de scales `[{"id": 0, "scale": 1.0}]` (non listé ⇒ scale 0.0), ou hot-reload `POST /lora-adapters`.
- Le **rôle** (juge/proc./avocat) = **system prompt** par requête.

> Caveat connu `llama.cpp` : des requêtes à **configs LoRA différentes ne sont pas batchées**. **Non-problème** ici : jeu solo, tours séquentiels.

---

## 9. Décisions clés & justifications (récap)

| Décision | Choix | Raisonnement | Alternative écartée |
|---|---|---|---|
| Déploiement | **100 % local, llama.cpp** | Hors-ligne, zéro coût d'API, on-device | Cloud / hybride |
| Modèle acteurs | **MiniCPM5-1B** | Archi Llama ⇒ LoRA+GGUF friendly ; SOTA-1B ; GGUF officiel | Qwen3-1.7B, Gemma3-1B (candidats valides à benchmarker) |
| Modèle Game Master | **Nemotron 3 Nano 4B (vanilla)** | Raisonnement fort, ciblé « gaming NPCs » edge, GGUF/llama.cpp ; hybride OK car **sans LoRA** | GM en 1B (planification trop faible) |
| Teacher données | **Gemma 4 31B-it, hors-ligne** | Qualité de distillation ; ne tourne pas chez le joueur | Teacher ≤ student (qualité insuffisante) |
| Jargon | **1 LoRA par métier** | Ajout d'un métier = 1 adaptateur ; 1 seul chargé/partie | 4 modèles fine-tunés (4× VRAM, dérive) |
| Rôle | **System prompt par rôle** | Gratuit, pas de poids dédiés | 1 LoRA par rôle (inutilement lourd) |
| Curriculum | **Stage juridique → fork jargon (checkpoint-fork)** | Base **vanilla** ⇒ llama.cpp = 1 base + N adaptateurs + GGUF officiel | Merge dans la base (base custom, couplage) |
| Orchestrateur | **Agentique (GM) + loquet déterministe** | Déterministe pur = répétitif ; aléatoire pur = chaos. Agentique **dans** des contraintes dures | Machine à états figée / RNG brut |
| Scénariste + Directeur | **Re-mergés en un Game Master 4B** | L'auteur des indices révèle le mieux ses indices ; 4B capable ; partie courte ⇒ coût borné | Split (rationnel ressources, mais handoff/cohérence) |
| Vision long terme | **Budget de tours + calendrier d'indices** | Partie courte ⇒ ordonnancement, pas planification profonde | Arc dramatique profond pré-écrit (sur-engineering) |
| Sortie du GM | **JSON contraint par GBNF** | Petit modèle : classe bien, invente mal ⇒ choix dans un deck | Prose libre (dérive) |
| Glossaire | **Tester sans d'abord** | Modèles récents compétents ; fallback glossaire si besoin (cohérence, difficulté, scoring) | Glossaire obligatoire dès le départ |

---

## 10. Points ouverts / à vérifier

- [ ] **Qualité LoRA sur base quantifiée** (Q4 vs Q8) côté MiniCPM5 dans llama.cpp.
- [ ] **Valeur du budget de tours** `B` et mapping difficulté ↔ (nb d'indices, obscurité du jargon, B).
- [ ] **Framework d'entraînement** TRL+PEFT (contrôle programmatique du fork) vs Unsloth (le plus rapide). Choix réversible.
- [ ] **Distiller un adaptateur directeur** : non nécessaire en v1 (GM = 4B vanilla + GBNF). À reconsidérer seulement si besoin.
- [ ] **Budget mémoire** sur la cible matérielle réelle (les deux modèles + adaptateur).

---

## 11. Stack & fichiers (implémentés)

**Runtime — `buzzwords/`** (app Gradio 100 % locale) :
- `config.py` — chemins des poids, styles de jargon, budget de tours, `REQUIRED_MODELS`.
- `models.py` — `CaseFile` / `GMDecision` / `Line` / `Case` / `GameSession`.
- `text_engine.py` — llama.cpp : Game Master (Nemotron, GBNF) + acteurs (MiniCPM + LoRA de style) ; grammaires GBNF Case File / décision / score.
- `pipeline.py` — écriture du Case File, boucle de tours (§5), scoring, `preflight()`.
- `tts_engine.py` — VoxCPM2 optionnel (`@spaces.GPU` ; fallback texte).
- `scene.py` / `ui.py` / `theme.py` — scène HTML + UI **`gr.Walkthrough`** (phases → étapes) + CSS.

**Entraînement hors-ligne — `training/`** (Modal GPU) :
- `teacher_datagen.py` — teacher Gemma 4 **FP8** (vLLM, L40S) → données ShareGPT, prompts seedés + filtrage (§7.1).
- `finetune.py` — curriculum LoRA 2 étapes, checkpoint-fork, sur MiniCPM5 (§7.3).
- `convert_gguf.py` — adaptateurs PEFT → GGUF (`convert_lora_to_gguf.py`, sans merge ; §7.5).
- `README.md` — ordre d'exécution + auth Modal/HF.

**Outils** : Modal (entraînement + génération hors-ligne), `llama.cpp` (inférence locale + LoRA GGUF + GBNF), Unsloth (TRL `SFTTrainer` + PEFT), `convert_lora_to_gguf.py` (conversion).

---

## 12. Faits vérifiés & sources

- **MiniCPM5-1B** : archi `LlamaForCausalLM`, Apache-2.0, 131K ctx, GGUF + mode thinking, SOTA-1B. Unsloth + 4 autres frameworks documentés par OpenBMB.
  - https://huggingface.co/openbmb/MiniCPM5-1B · https://github.com/openbmb/minicpm
- **Gemma 4 31B-it** : open weights, 256K ctx, multilingue (≈ avril 2026).
  - https://huggingface.co/google/gemma-4-31B-it · https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- **Nemotron 3 Nano 4B** : hybride Mamba-2 (21/4/17), 262K ctx, GGUF officiel + Unsloth, llama.cpp (Q4_K_M ~18 tok/s Jetson 8 Go), ciblé « AI gaming NPCs » (mars 2026).
  - https://huggingface.co/blog/nvidia/nemotron-3-nano-4b · https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF · https://github.com/ggml-org/llama.cpp/discussions/20421
- **llama.cpp multi-LoRA** : `--lora`, `--lora-init-without-apply`, scales par requête, `POST /lora-adapters` ; `convert_lora_to_gguf.py` (PEFT→GGUF, sans merge).
  - https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md · https://github.com/ggml-org/llama.cpp/pull/10994 · https://huggingface.co/blog/ngxson/gguf-my-lora

---

*État : runtime (boucle GBNF + UI `gr.Walkthrough`) et scaffolds d'entraînement Modal implémentés ; reste à entraîner les adaptateurs et brancher les poids GGUF réels. Voir le mapping design → code en §11.*
