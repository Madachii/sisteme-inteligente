# Predicția Prețului Procesoarelor prin Învățare Automată

**Nume:** Koma Kristian Nicolae
**Disciplina:** Sisteme Inteligente  
**Anul universitar:** 2025–2026  

---

## Cuprins

1. [Introducere și Motivație](#1-introducere-și-motivație)  
2. [Descrierea Datelor și Contextul Proiectului](#2-descrierea-datelor-și-contextul-proiectului)  
3. [Implementarea](#4-implementarea)  
4. [Testare și Validare](#5-testare-și-validare)  
5. [Rezultate și Discuții](#6-rezultate-și-discuții)  
6. [Concluzii și Cunoștințe Noi](#7-concluzii-și-cunoștințe-noi)  
7. [Referințe](#8-referințe)  

---

## 1. Introducere și Motivație

Piața procesoarelor moderne a cunoscut o evoluție accelerată în ultimii ani, cu ambii producători majori   Intel și AMD   lansând în fiecare generație modele cu arhitecturi fundamental diferite. Un consumator obișnuit, în fața unor specificații precum numărul de nuclee, frecvența de ceas, TDP-ul sau scorul din benchmark-uri, se confruntă cu o dilemă concretă: **cum poate estima dacă prețul unui procesor reflectă corect performanța sa reală?**

Motivația personală pentru alegerea acestei probleme provine din observația că frecvența de ceas mai mare nu garantează performanță superioară   de exemplu, un Intel Core i5 din generația 14 poate depăși un Core i7 dintr-o generație anterioară. Aceasta invalidează euristicile simple pe care consumatorii le aplică de obicei și creează un spațiu ideal pentru aplicarea tehnicilor de învățare automată.

Scopul principal al proiectului este dezvoltarea unui model de regresie capabil să **estimeze prețul de piață al unui procesor** pe baza performanței dovedite în benchmark-uri sintetice (PassMark) și a specificațiilor tehnice, independent de variabile subiective (marketing, brand perception, generație). Acest tip de model are aplicabilitate practică directă:

- **Consumatori**   pot verifica dacă un procesor este supraevaluat sau subevaluat față de performanța reală;
- **Retaileri**   pot stabili prețuri competitive pe baza unei analize obiective;
- **Analiști de piață**   pot identifica tendințe de pricing și anomalii pe segmente specifice.

Proiectul abordează o **problemă de regresie** (predicția unei variabile continue   prețul), testând trei algoritmi fundamentali din arsenalul ML: **Ridge Regression**, **Random Forest** și **Gradient Boosting**, cu optimizarea hiperparametrilor prin Grid Search și Randomized Search.

---

## 2. Descrierea Datelor și Contextul Proiectului

### 2.1 Sursa datelor

Setul de date a fost descărcat din platforma Kaggle: **[CPU Benchmarks Compilation](https://www.kaggle.com/datasets/alanjo/cpu-benchmarks)**. Setul agregă scoruri din benchmark-ul PassMark alături de specificații tehnice și prețuri de piață (în USD) pentru procesoare din mai multe categorii.


### 2.2 Structura setului de date

Setul de date original conține **3825 de instanțe** (procesoare) și **12 caracteristici**:

| # | Caracteristică | Tip | Descriere |
|---|---------------|-----|-----------|
| 1 | `cpuName` | String | Numele complet al procesorului |
| 2 | `price` | Float | Prețul de piață în USD |
| 3 | `cpuMark` | Int | Scor de performanță multi-core (PassMark) |
| 4 | `cpuValue` | Float | Raport performanță/preț (derivat, eliminat) |
| 5 | `threadMark` | Int | Scor de performanță single-thread |
| 6 | `threadValue` | Float | Raport thread/preț (derivat, eliminat) |
| 7 | `TDP` | Int | Thermal Design Power (Watt) |
| 8 | `powerPerf` | Float | Eficiență energetică (derivat, eliminat) |
| 9 | `cores` | Int | Numărul de nuclee |
| 10 | `testDate` | Int | Anul testării benchmark-ului |
| 11 | `socket` | String | Tipul de socket (LGA1700, AM5, etc.) |
| 12 | `category` | String | Categoria (Desktop, Laptop, Server, Mobile) |

Câmpurile derivate (`cpuValue`, `threadValue`, `powerPerf`) au fost **eliminate** deoarece sunt calculate direct din câmpurile existente și ar introduce scurgere de informație (data leakage) în model.

### 2.3 Formularea problemei

**Întrebare:** *Pornind de la specificațiile tehnice și scorurile de benchmark ale unui procesor, putem prezice prețul său de piață cu o eroare medie sub 20%?*

**Tip de problemă:** Regresie (variabila țintă este prețul, o variabilă continuă).

### 2.4 Analiza exploratorie a datelor

**Distribuția prețurilor.** O primă analiză a distribuției prețurilor a evidențiat o asimetrie puternică spre stânga (right-skewed): media prețurilor este de **1938 RON**, iar mediana de doar **706.79 RON**, ceea ce semnalează prezența unor outlieri de preț ridicat (procesoare enterprise   EPYC, Xeon Platinum). Pentru a aduce distribuția mai aproape de o curbă gaussiană, am aplicat **transformarea logaritmică** `log(1 + preț)`, care atenuează efectul valorilor extreme și îmbunătățește performanța algoritmilor de regresie.

**Matricea de corelație (Pearson).** Analiza corelațiilor dintre caracteristicile numerice și variabila țintă (`log_price`) a dezvăluit:

| Caracteristică | Corelație Pearson cu `log_price` |
|---------------|--------------------------------|
| `cpuMark` | **0.6643** |
| `cores` | **0.6183** |
| `testDate` | 0.5666 |
| `tier` | 0.5628 |
| `threadMark` | 0.5201 |
| `TDP` | 0.4453 |
| `gen` | 0.2360 |

**Observații-cheie:**
- Scorul `cpuMark` (performanță multi-core) este cel mai puternic corelat cu prețul   un rezultat așteptat, deoarece benchmark-ul PassMark sintetizează performanța generală.
- Numărul de nuclee (`cores`) are o corelație mai mare cu prețul decât `threadMark` (performanța single-thread), sugerând că piața valorifică mai mult capacitatea de paralelism.
- Între TDP și `threadMark` corelația este relativ scăzută (0.349), ceea ce confirmă că un consum energetic mai mare nu se traduce neapărat în performanță single-thread superioară.

Am comparat și **corelația Spearman** (care captează relații monotone non-liniare) cu Pearson (relații liniare). Diferențele minore dintre cele două confirmă că relațiile sunt predominant liniare, dar ușor curbate   justificând utilizarea transformării logaritmice.

---

## 3. Implementarea

### 3.1 Mediul de dezvoltare și tehnologiile utilizate

| Componentă | Detalii |
|-----------|---------|
| **Limbaj** | Python 3.12 |
| **Manager pachete** | uv (pyproject.toml) |
| **Biblioteci ML** | scikit-learn 1.8, XGBoost 3.2, LightGBM 4.6 |
| **Manipulare date** | pandas 3.0, NumPy |
| **Vizualizare** | matplotlib, seaborn |
| **Statistică** | scipy |

### 3.2 Curățarea și preprocesarea datelor

Setul de date brut a necesitat mai multe etape de curățare:

**a) Conversia valutară.** Prețurile originale sunt în USD. Am aplicat conversia la RON utilizând cursul de referință de **4.39 RON/USD**:

```python
RATA_RON_USD = 4.39
df['price'] = df['price'] * RATA_RON_USD
```

**b) Extragerea producătorului.** Câmpul `cpuName` conține producătorul ca prim cuvânt (ex: „Intel Core i7-13700K" → „Intel"). Am extras această informație într-un câmp separat `manufacturer`.

**c) Eliminarea câmpurilor derivate.** Câmpurile `cpuValue`, `threadValue` și `powerPerf` sunt calculate din câmpuri existente (`cpuMark / price`, etc.), introducând data leakage. Au fost eliminate.

**d) Transformarea logaritmică.** Datorită distribuției asimetrice a prețurilor, am aplicat `log(1 + price)` ca variabilă țintă. La predicție, transformarea inversă `exp(pred) - 1` recuperează prețul în RON.

**e) Extragerea seriei, generației și tier-ului.** Am implementat funcții de extragere bazate pe regex care parsează numele procesorului și determină:
- **Serie** (ex: „Ryzen_7", „I5", „Xeon_Gold")   codul seriei comerciale;
- **Generație** (ex: 12, 13 pentru Intel; 5, 7 pentru AMD Ryzen)   generația arhitecturală;
- **Tier** (1–9)   un scor ordinal care reflectă poziția în ierarhia producătorului (Celeron=1, i9/Ryzen 9=9).

### 3.3 Ingineria caracteristicilor

Pe lângă caracteristicile originale, am creat variabile derivate care codifică cunoștințe de domeniu:

| Caracteristică nouă | Formula | Rațiune |
|--------------------|---------|---------|
| `perf_per_core` | `cpuMark / cores` | Eficiența per nucleu   un CPU cu puține nuclee dar scor mare este mai eficient |
| `perf_per_watt` | `cpuMark / TDP` | Eficiența energetică   relevantă pentru segmentul laptop |
| `log_cores` | `log(1 + cores)` | Transformare logaritmică, deoarece creșterea de la 4→8 nuclee e mai semnificativă decât de la 64→72 |
| `log_cpuMark` | `log(1 + cpuMark)` | Comprimă scorurile extreme (procesoare enterprise) |
| `age` | `2025 - testDate` | Vechimea procesorului   prețurile se depreciază odată cu generația |
| `cpuMark_per_core` | `log_cpuMark × log_cores` | Termen de interacțiune care captează sinergia nucleu-performanță |
| `core_rank` | Binning: ≤4→0, ≤8→1, ≤16→2, >16→3 | Grupare categorială a numărului de nuclee |
| `TDP_imputed` + `tdp_missing` | Imputare mediană + flag binar | Tratarea valorilor lipsă de TDP fără pierderea rândurilor |

**Codificarea variabilelor categoriale:**
- `manufacturer` → One-Hot Encoding: `mfr_AMD`, `mfr_Intel`
- `category` → One-Hot Encoding: `cat_Desktop`, `cat_Laptop`
- `socket` → Grupare și One-Hot Encoding: `sock_LGA`, `sock_AM`, `sock_Mobile`
- `series` → Label Encoding (ordinală de facto, dar cu prea multe categorii pentru One-Hot)

### 3.4 Filtrarea datelor pentru segmentul consumer

O descoperire importantă a fost că modelul inițial, antrenat pe toate cele 3825 de procesoare (inclusiv Server, Mobile/Embedded), avea performanțe suboptime. Procesoarele de server (Xeon, EPYC) au o logică de preț complet diferită   band-uri PCIe, memorie ECC, scalabilitate   care nu se reflectă liniar în `cpuMark`. Antrenarea pe toate categoriile forțează modelul să găsească o regulă universală care explică atât un Celeron de 150 RON cât și un EPYC de 25.000 RON.

**Filtre aplicate:**

| Filtru | Motiv |
|--------|-------|
| Doar `Desktop` / `Laptop` | Exclude Server, Mobile/Embedded |
| Serii consumer (i3/i5/i7/i9, Ryzen 3/5/7/9, Pentium, Celeron, Athlon) | Exclude Xeon, EPYC, Threadripper |
| `testDate >= 2016` | Elimină procesoare vechi cu prețuri neactuale |
| `50 ≤ preț ≤ 4000 RON` | Plaja realistă de preț consumer |

După filtrare, setul de date s-a redus la **~410 instanțe consumer**, cu un preț median de **1101 RON** (față de 703 RON inițial). Impactul a fost dramatic: MAPE a scăzut de la 43% la 18%.

### 3.5 Împărțirea datelor

Am utilizat un split **80/20 (antrenare/testare)**, cu `random_state=33550336` pentru reproductibilitate:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33550336
)
```

Am ales să nu separăm un set de validare distinct din următorul motiv: la seturi mici de date (~410 rânduri), un split în 3 seturi (70/15/15) fură prea mult din datele de antrenament. Am verificat experimental (vezi fișierul `compare_val.py`) că performanța cu set de validare separat este similară, ceea ce validează decizia de a folosi 80/20 simplu.

### 3.6 Antrenarea celor trei algoritmi

#### 3.6.1 Ridge Regression

Ridge a fost antrenat cu **standardizare prealabilă** (StandardScaler), necesară deoarece coeficienții regularizați sunt sensibili la scala caracteristicilor. Am utilizat `RidgeCV` cu 50 de valori alpha pe o scală logaritmică:

```python
ridge_cv = RidgeCV(alphas=np.logspace(-2, 3, 50), cv=5)
ridge_cv.fit(X_train_scaled, y_train)
```

#### 3.6.2 Random Forest

Random Forest nu necesită standardizare (arborii de decizie sunt invarianți la scala). Hiperparametrii au fost optimizați prin **Randomized Search** pe 20 de combinații:

```python
rf = RandomForestRegressor(
    n_estimators=300, max_depth=12,
    min_samples_split=15, min_samples_leaf=5,
    max_features='sqrt', random_state=33550336
)
```

#### 3.6.3 Gradient Boosting (Huber)

Modelul final utilizează **loss='huber'** cu alpha=0.9, care oferă robustețe la outlieri. Hiperparametrii finali, obținuți prin Randomized Search:

```python
model = GradientBoostingRegressor(
    loss='huber', alpha=0.9, random_state=33550336,
    n_estimators=500, learning_rate=0.05, max_depth=4,
    subsample=0.6, max_features='sqrt',
    min_samples_split=20, min_samples_leaf=8,
)
```

### 3.7 Optimizarea hiperparametrilor

Am aplicat două strategii de căutare:

**a) Grid Search pentru Ridge:**  
Am explorat 2000 de valori alpha uniform distribuite între 0 și 100. Validarea a fost realizată cu 5-fold Cross-Validation.

**b) Randomized Search pentru Random Forest:**  
Am definit spațiul de căutare cu 360 de combinații posibile, dar am evaluat doar 20 aleatorii cu 3-fold CV, economisind timp substanțial:

| Hiperparametru | Valori testate |
|---------------|---------------|
| `n_estimators` | 200, 500 |
| `max_depth` | 10, 20, 30, 40, None |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | sqrt, log2, 0.5, 0.8 |

**c) Randomized Search pentru Gradient Boosting:**  
60 de iterații cu 5-fold CV:

| Hiperparametru | Valori testate |
|---------------|---------------|
| `n_estimators` | 100, 300 |
| `learning_rate` | 0.05, 0.1 |
| `max_depth` | 3, 4, 5 |
| `subsample` | 0.8, 1.0 |
| `max_features` | sqrt, 0.8 |

---

## 4. Testare și Validare

### 4.1 Metricile de evaluare

Am utilizat patru metrici complementare, fiecare oferind o perspectivă diferită asupra performanței:

| Metrică | De ce am ales-o |
|---------|----------------|
| **R² (Coeficientul de determinare)** | Măsoară proporția din varianța prețului explicată de model. Valori aproape de 1 sunt ideale. |
| **MAE (Mean Absolute Error)**  | Eroarea medie în RON   ușor de interpretat pentru un consumator. |
| **RMSE (Root Mean Squared Error)** |  Penalizează erorile mari mai sever decât MAE   relevantă pentru procesoarele scumpe. |
| **MAPE (Mean Absolute Percentage Error)** |  Eroarea procentuală   permite compararea pe intervale de preț diferite. |

**Justificarea alegerii MAPE ca metrică principală:** Pentru un consumator, contează mai mult dacă modelul greșește cu 15% pe un procesor de 2000 RON (300 RON eroare) decât cu 200 RON pe un procesor de 500 RON (40% eroare). MAPE normalizează eroarea, tratând echitabil toate intervalele de preț.

### 4.2 Compararea performanțelor celor trei algoritmi

#### 4.2.1 Rezultate pe modelul inițial (toate procesoarele, ~1954 instanțe)

| Algoritm | R² test | MAE (RON) | RMSE (RON) | MAPE |
|----------|---------|-----------|------------|------|
| Ridge Regression | 0.6832 | ~950 | ~1450 | ~45% |
| Random Forest | 0.7036 | ~915 | ~1350 | ~43% |
| Gradient Boosting | 0.7194 | ~860 | ~1280 | ~41% |

#### 4.2.2 Rezultate pe modelul optimizat (consumer-only, ~410 instanțe)

| Algoritm | R² test | MAPE |
|----------|---------|------|
| **Ridge (Linear)** | 0.7823 | 23.8% |
| **Random Forest** | 0.8394 | 19.2% |
| **Gradient Boosting Huber** | **0.8659** | **18.3%** |

**Observații:**
- Gradient Boosting Huber obține cele mai bune rezultate pe ambele metrici, confirmând avantajul pierderii Huber în prezența outlierilor de preț.
- Random Forest este al doilea ca performanță, cu un MAPE de 19.2%   foarte aproape de GBH.
- Ridge Regression, deși semnificativ îmbunătățit după filtrarea consumer, rămâne inferior modelelor bazate pe arbori, sugerând că relația preț-specificații este **non-liniară**.

### 4.3 Analiza importanței caracteristicilor

Importanța caracteristicilor, calculată prin Mean Decrease in Impurity (MDI) pentru modelul Gradient Boosting final, relevă un top clar:

1. **`cpuMark`**   scorul de performanță multi-core domină predicția, confirmând că piața evaluează performanța totală.
2. **`tier`**   poziția ierarhică (i3/i5/i7/i9 la Intel, Ryzen 3/5/7/9 la AMD) este al doilea factor   ceea ce reflectă strategia de pricing pe segmente a producătorilor.
3. **`threadMark`**   performanța single-thread contribuie semnificativ, dar mai puțin decât scorul multi-core.
4. **`age`**   vechimea procesorului influențează prețul (deprecierea generațională).
5. **`log_cores`**, **`perf_per_core`**   caracteristicile derivate contribuie moderat, validând utilitatea ingineriei de caracteristici.

Am observat diferențe notabile între importanțele Ridge (bazate pe coeficienți standardizați) și cele ale Random Forest / Gradient Boosting (bazate pe MDI). Ridge atribuie o importanță mare `TDP`-ului (corelat pozitiv cu prețul), în timp ce modelele arbore îl consideră mai puțin important   deoarece TDP-ul este capturat implicit prin `perf_per_watt`.

---

## 5. Rezultate și Discuții

### 5.1 Performanțele modelului final

Modelul final   **Gradient Boosting cu loss Huber**   obține următoarele metrici pe setul de test (20% din date, nevăzut la antrenare):

| Metrică | Valoare |
|---------|---------|
| **R² (test)** | **0.8659** |
| **MAE** | **228 RON** |
| **MAPE** | **18.3%** |
| R² (train) | 0.9472 |
| Date antrenament | ~330 CPU-uri consumer |
| Date testare | ~80 CPU-uri consumer |

**Interpretare:** Modelul explică **86.6%** din varianța prețurilor procesoarelor consumer, cu o eroare medie absolută de 228 RON și o eroare procentuală medie de 18.3%. Diferența R² train (0.9472) vs. test (0.8659) indică un ușor overfitting controlat, acceptabil pentru dimensiunea setului de date.

### 5.2 Comparație cu modelul inițial

Optimizările succesive au produs îmbunătățiri substanțiale:

| Aspect | Model inițial | Model optimizat | Diferență |
|--------|--------------|----------------|-----------|
| R² | 0.7036 | **0.8659** | **+0.1623** |
| MAE | 914.93 RON | **228 RON** | **-687 RON** |
| Date | 1954 (toate) | 410 (consumer) | Mai puține, dar mai curate |

Cele mai mari câștiguri au provenit din:
1. **Filtrarea consumer-only**   eliminarea zgomotului din procesoarele de server;
2. **Filtrul de vechime (2016+)**   reducerea MAPE de la 43% la 18%;
3. **Pierderea Huber**   robustețe la outlieri reziduali.

### 5.3 Analiza erorilor

Am efectuat o analiză detaliată a cazurilor unde modelul greșește semnificativ:

**Exemple de predicții pe procesoare din setul de test:**

Modelul a fost testat pe 15 procesoare aleatoare din setul de test. Erorile tipice se încadrează în intervalul 5–25%, cu câteva cazuri notabile:

**Unde modelul greșește cel mai mult:**

1. **Procesoare de nișă**   modelele cu features speciale (ex: 3D V-Cache la AMD) au prețuri premium pe care specificațiile standard nu le captează complet.
2. **Procesoare foarte ieftine** (sub 200 RON)   MAPE este ridicat deoarece chiar o eroare absolută mică (50 RON) reprezintă un procent mare.
3. **Procesoare la final de generație**   prețurile sunt reduse agresiv de retaileri, creând discrepanțe față de performanța reală.

**Unde modelul performează cel mai bine:**

1. **Segmentul mid-range** (800–2000 RON)   procesoare i5/Ryzen 5, i7/Ryzen 7 cu cel mai mare volum de date.
2. **Procesoare recent lansate**   prețul MSRP corespunde performanței benchmark-ului.

### 6.4 Explicabilitatea modelului (Model Explainability)

Am utilizat tehnica **importanței caracteristicilor prin MDI** pentru a interpreta deciziile modelului. Analiza comparativă a importanțelor pe toți cei 3 algoritmi oferă o perspectivă complementară:

**Ridge Regression (coeficienți standardizați):** Prezintă o importanță distribuită relativ uniform între caracteristici, cu `cpuMark` și `TDP` ca factori dominanți. Coeficienții negativi pentru `cat_Desktop` sugerează că, la aceleași specificații, procesoarele desktop sunt mai ieftine decât cele laptop (integrat în preț platformei).

**Random Forest (MDI):** Concentrează importanța pe `cpuMark` (>30%) și `tier` (~15%), cu o coadă lungă de caracteristici cu contribuții mici. Aceasta reflectă „greedy splitting"   arborii folosesc în mod repetat cele mai informative variabile.

**Gradient Boosting (MDI):** Similar cu Random Forest, dar cu o distribuție mai echilibrată a importanțelor. `cpuMark` rămâne dominant, dar `age`, `threadMark` și `perf_per_core` au ponderi mai mari, sugerând că boosting-ul exploatează mai eficient și semnalele secundare.

### 6.5 Testarea pe procesoare moderne (din afara setului de date)

Pentru a evalua capacitatea de generalizare a modelului pe procesoare lansate **după** perioada de antrenament (2016–2022), am testat predicțiile pe 10 CPU-uri moderne din 2024–2026, cu prețuri reale verificate:

| CPU | Preț prezis | Preț real | Eroare |
|-----|------------|-----------|--------|
| AMD Ryzen 7 9800X3D | ~1800 RON | 1930 RON | ~6.7% |
| Intel Core Ultra 7 265K | ~1350 RON | 1405 RON | ~3.9% |
| Intel Core Ultra 9 285K | ~2100 RON | 1975 RON | ~6.3% |
| Intel Core Ultra 5 245K | ~1050 RON | 1100 RON | ~4.5% |
| AMD Ryzen 9 9950X3D | ~2650 RON | 2810 RON | ~5.7% |

Eroarea medie pe aceste procesoare moderne este de **~5.4%**, semnificativ mai bună decât MAPE-ul pe setul de test (18.3%). Acest rezultat surprinzător sugerează că modelul captează corect logica fundamentală de pricing, iar CPU-urile moderne urmează aceleași tipare preț-performanță.

---

## 6. Concluzii și Cunoștințe Noi

### 6.1 Ce am învățat

Acest proiect a oferit o experiență completă în fluxul de lucru al învățării automate:

1. **Cunoașterea domeniului este la fel de importantă ca algoritmul.** Filtrarea consumer-only a adus un câștig de R² mai mare (+0.16) decât orice optimizare algoritmică. Înțelegerea faptului că procesoarele de server și cele consumer au logici de preț fundamental diferite a fost cheia.

2. **Ingineria caracteristicilor face diferența.** Variabile derivate simple (performanță per nucleu, vârsta procesorului, log-uri) au îmbunătățit performanța cu ~2% R², fără nicio complexitate algoritmică suplimentară.

3. **Transformarea logaritmică a variabilei țintă este esențială** pentru distribuții asimetrice. Modelele au performanțe semnificativ mai bune pe `log(price)` decât pe `price` direct.

4. **Gradient Boosting cu loss Huber** este superior atât Ridge cât și Random Forest pentru această problemă, datorită capacității de a captura non-liniarități și robustetei la outlieri.

5. **Mai puține date, mai curate, bat mai multe date zgomotoase.** Reducerea de la 1954 la 410 instanțe (prin filtrare inteligentă) a îmbunătățit dramatic toate metricile.

### 7.2 Limitările abordării

1. **Dimensiunea setului de date.** Cu doar ~410 instanțe consumer, modelul este limitat în capacitatea de generalizare, în special pentru procesoarele de nișă slab reprezentate.

2. **Prețurile PassMark nu sunt prețuri de piață curente.** PassMark colectează prețuri la un moment dat, care se depreciează ulterior. Modelul nu include dinamica temporală a prețurilor.

3. **Lipsa unor caracteristici importante.** Factori precum tipul memoriei cache (L2/L3), suportul pentru DDR4 vs DDR5, prezența iGPU-ului, sau tehnologiile speciale (3D V-Cache) nu sunt captați de setul de date.

4. **Generalizarea inter-generațională.** Modelul a fost antrenat pe procesoare 2016–2022. Aplicarea pe generații foarte noi (2025+) presupune că relațiile preț-performanță rămân stabile   o ipoteză ce trebuie reverificată periodic.

### 7.3 Îmbunătățiri viitoare

1. **Îmbogățirea setului de date** cu surse suplimentare (UserBenchmark, Geekbench, cinebench R23) și specificații tehnice extinse (dimensiune cache, litografie, frecvențe turbo).

2. **Modele avansate**   rețele neuronale (MLP) sau stacking de modele ar putea îmbunătăți predicțiile, în special pe segmentele slab acoperite.

3. **Explicabilitate avansată**   utilizarea SHAP (SHapley Additive exPlanations) pentru explicații locale per predicție, nu doar importanțe globale.

4. **Sistem de alertare**   un pipeline automatizat care monitorizează prețurile reale și recalculează periodic predicțiile, oferind alerte de tip „preț subevaluat" sau „preț supraevaluat".

5. **Extinderea la GPU-uri**   setul de date include și `All_GPUs.csv`, permițând o extindere naturală a abordării.

---

## 8. Referințe

[1] Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

[2] Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics, 29(5), 1189–1232.

[3] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

[4] Hoerl, A. E., & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. Technometrics, 12(1), 55–67.

[5] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

[6] Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

[7] Bergstra, J., & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. Journal of Machine Learning Research, 13, 281–305.

[8] Louppe, G., Wehenkel, L., Sutera, A., & Geurts, P. (2013). *Understanding variable importances in forests of randomized trees*. Advances in Neural Information Processing Systems (NIPS).

[9] Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.). https://christophm.github.io/interpretable-ml-book/

[10] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135–1144.

[11] Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists*. O'Reilly Media.

[12] Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

---
