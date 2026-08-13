"""Tab content for the system walkthrough. Imported by tools.walkthrough.build.

Every figure here is either (a) quoted from the v3 deck / docs/design.md, i.e.
the baseline-20260811043259 production run, or (b) computed from this repo's
own code with the inputs stated inline, so a reader can re-run it. Where a
figure is a constructed illustration rather than a measurement, it says so.
"""


def filecard(path, holds, state, reader, moves=False):
    tag = "moves-tag" if moves else ""
    return f"""  <div class="filecard">
    <b>{path}</b>
    <span><i>holds</i>{holds}</span>
    <span><i>read by</i>{reader}</span>
    <span class="frozen-tag {tag}">{state}</span>
  </div>"""


# ================================================================= 0 · the map
P_MAP = f"""
<section class="prose" style="padding-top:34px">
  <div class="step-head"><h2>Freeze what history can support. Learn the one thing it cannot.</h2></div>
  <p class="lede">
    Five months of history tells us accurately <em class="term">how much of a thing sells
    in a given hour</em>. It cannot tell us <em class="term">what happens if we change the
    price</em> — and not because the data is thin. Two specific confounds make that
    question unanswerable from records of a policy that was never randomised.
  </p>
</section>

<figure style="margin-top:8px">
  <div class="scroller"><table class="tl">
    <caption>The two confounds everything else is built around</caption>
    <thead><tr><th>Confound</th><th>What it does to a naive estimate</th></tr></thead>
    <tbody>
      <tr>
        <td><strong>The clock</strong></td>
        <td>The legacy rule discounts by time elapsed, so deep discounts happen late — and late is also when the store is busy. Fit price against sales and you recover the evening rush wearing a price label. The bias points toward <em>too elastic</em>: it looks like discounting works far better than it does.</td>
      </tr>
      <tr>
        <td><strong>Survivorship</strong></td>
        <td>A row at a deep discount exists <em>because the earlier hours did not sell</em>. Deep-discount rows are therefore drawn from the slow-moving tail of every episode. Condition on that and demand looks unresponsive. The bias points the other way — toward <em>too inelastic</em> — and it is strongest exactly where we most want to know.</td>
      </tr>
    </tbody>
  </table></div>
  <figcaption>
    Two confounds pointing in opposite directions, both large. That is why the answer is
    not a better model on the same data, and why the elasticity tab is the one worth
    arguing about.
  </figcaption>
</figure>

<section class="prose">
  <p style="margin-top:26px">
    So the system is split. Everything history genuinely supports is measured once and
    pinned in a file. The single quantity it cannot support is given a wide, honest
    starting guess and then learned in production, from prices we randomise on purpose.
  </p>
</section>

<figure style="margin-top:8px">
  <div class="scroller"><table class="lefty">
    <caption>Everything the agent stands on</caption>
    <thead><tr><th>Artifact</th><th>What it holds</th><th>State</th></tr></thead>
    <tbody>
      <tr><td>split_manifest.json</td><td>the analysis population — 13 counted filter stages</td><td class="good">frozen</td></tr>
      <tr><td>baseline_model.txt</td><td>μ_ref — units next hour at the reference discount</td><td class="good">frozen</td></tr>
      <tr><td>calibration.json</td><td>one level factor per subcategory</td><td class="good">frozen</td></tr>
      <tr><td>r_lookup.json</td><td>demand dispersion, by subcategory</td><td class="good">frozen</td></tr>
      <tr><td>rho.json</td><td>within-episode correlation, and the evidence tax it implies</td><td class="good">frozen</td></tr>
      <tr><td>prior.json</td><td>the cold-start price response — −1.0 ± 0.6</td><td class="good">frozen</td></tr>
      <tr><td>posterior.json</td><td>what production has learned about price response</td><td class="loss">the only one that moves</td></tr>
      <tr><td>pricing/dp.py</td><td>nothing — it holds no state, and solves from scratch each hour</td><td class="good">code</td></tr>
    </tbody>
  </table></div>
</figure>

<div class="callout">
  <h3>Frozen is a phase, not a doctrine</h3>
  <p>
    Freezing buys one specific thing: <strong>a fixed yardstick for the MVP and the
    experiment</strong>. If the demand model moved while we were measuring whether a
    pricing change helped, we could not say which of the two caused the difference — and
    the honest answer would be that we no longer know. So for this phase the rule is
    absolute, and every report stamps the model version it ran against so the check is
    mechanical rather than remembered.
  </p>
  <p style="margin-top:14px">
    <strong>Once the experiment has read out, that constraint lifts.</strong> The baseline
    moves to a regular retraining schedule like any other production model, and the level
    factors track drift between retrains. Nothing in the architecture depends on the model
    staying still — only the <em>measurement</em> does.
  </p>
</div>

<section class="prose">
  <h3 style="margin-top:34px">The one that moves</h3>
  <p>
    <code>posterior.json</code> holds the belief about price response, and it is the only
    thing production changes. It starts at the prior, updates only from outcomes where the
    price was genuinely randomised, and each update is bounded, human-gated, and consumed
    exactly once. Everything else here exists to make that one number learnable safely —
    the <strong>Learning</strong> tab is how it moves.
  </p>
</section>
"""

# ================================================================= 1 · data
P_DATA = f"""
<section class="prose" style="padding-top:34px">
  <div class="step-head"><h2>The population every other number is measured on</h2></div>
  <p class="lede">
    Before anything is modelled, the raw feed is reduced to a population we are willing to
    defend, through <strong>thirteen named filter stages</strong>. Every stage is counted
    and written to the manifest — so any figure anywhere in this system traces back to the
    rows behind it.
  </p>
</section>

{filecard("artifacts/split_manifest.json", "the filter waterfall, the episode rule, and the train/calib/test split", "frozen", "every downstream step")}

<section class="prose">
  <p style="margin-top:24px">
    Structural drops go first — duplicate hour rows, out-of-range discounts, negative
    quantities, null categories, zero base prices, negative windows. Then the economic
    ones: windows longer than 120 hours (a known source defect rather than a genuinely
    long window), prices already below cost, non-priceable rows, sales exceeding
    inventory. Nothing is dropped silently at any stage.
  </p>
</section>

<ul class="chips" style="margin-top:26px">
  <li><span class="k">Hourly rows</span><span class="v">2.07M</span></li>
  <li><span class="k">Episodes</span><span class="v">334k</span></li>
  <li><span class="k">Filter stages</span><span class="v">13</span></li>
  <li><span class="k">Window cap</span><span class="v">120 h</span></li>
</ul>

<div class="callout">
  <h3>The definition that had to be got right</h3>
  <p>
    An <em class="term">episode</em> is one SKU in one facility across one clearance
    window — and it is deliberately <strong>not keyed by calendar date</strong>. Clearance
    windows routinely run past midnight. A date key would split one economic episode into
    two, reset the price ceiling halfway through, and charge the carried-over inventory to
    scrap at the seam — inventing a loss that never happened and hiding the decision that
    mattered.
  </p>
  <p style="margin-top:14px">
    Instead an episode is a maximal run of consecutive hours over which the source's own
    countdown decrements by exactly one per hour.
  </p>
</div>

<section class="prose">
  <h3 style="margin-top:34px">One exclusion that is about training, not about capability</h3>
  <p>
    Episodes restocked mid-window are dropped whole from the analysis population. That is a
    <strong>training-data decision</strong>: a restocked window mixes two inventory pools,
    so the sales it records are not cleanly attributable to the position the agent was in,
    and it would blur the very quantities we are trying to measure.
  </p>
  <p>
    <strong>It is not a limitation of the agent.</strong> In production a restock is
    trivial — inventory is part of the state, the DP re-solves from scratch every hour, and
    a shelf that went from 2 units to 6 is simply a different starting position at the next
    decision. Nothing carries over that a restock would invalidate.
  </p>
</section>

<figure>
  <div class="scroller"><table class="tl">
    <caption>The split, fixed before any model was fit</caption>
    <thead><tr><th>Window</th><th>Dates</th><th>What it is for</th></tr></thead>
    <tbody>
      <tr><td>train</td><td>2026-03-01 → 07-13</td><td>fitting the demand model</td></tr>
      <tr><td>calibration</td><td>2026-07-14 → 07-27</td><td>fitting the level factors and dispersion</td></tr>
      <tr><td>test</td><td>2026-07-28 → 08-03</td><td>grading the result — and nothing else</td></tr>
      <tr><td>excluded</td><td>2026-04-25 → 06-03</td><td>a known disturbed period, removed whole</td></tr>
    </tbody>
  </table></div>
  <figcaption>
    The gate window and the fitting window must never overlap, or the gate grades its own
    fit. That constraint is enforced in code, not by convention.
  </figcaption>
</figure>
"""

# ================================================================= 2 · model
P_MODEL = f"""
<section class="prose" style="padding-top:34px">
  <div class="step-head"><h2>A demand model deliberately blind to price</h2></div>
  <p class="lede">
    This is the only machine-learned component in the system, and it answers exactly one
    question.
  </p>
</section>

<div class="math">μ_ref(x)  =  E[ units sold in the next hour | features x, discount = d_ref ]

           gradient-boosted trees, Tweedie objective, fit on 2.07M hourly rows
           d_ref = the category's reference discount (30% in the worked episode)</div>

<section class="prose">
  <p style="margin-top:16px">
    One number per SKU per hour, at <em>one</em> price — the category reference. Every
    economic quantity the agent computes is denominated in it. A Tweedie objective is the
    right family here: hourly perishable demand is mostly zero with an occasional burst,
    which is precisely what Tweedie is for.
  </p>
</section>

{filecard("artifacts/baseline_model.txt", "μ_ref — expected units next hour at the reference discount", "frozen at launch", "the DP, every hour")}

<figure style="margin-top:26px">
  <div class="scroller"><table class="tl">
    <caption>The ten features, and why each is allowed</caption>
    <thead><tr><th>Feature</th><th>What it carries</th></tr></thead>
    <tbody>
      <tr><td><code>sku_ref_sales_rate_30d</code></td><td>how fast this SKU normally moves — a trailing 30-day rate built <strong>only from anchor hours</strong>, and point-in-time so it never sees the future</td></tr>
      <tr><td><code>prior_episode_ref_sales_rate</code></td><td>the same signal from this SKU's previous clearance window, again anchor hours only</td></tr>
      <tr><td><code>hour_of_day</code></td><td>the daily demand shape — the evening rush, explicitly</td></tr>
      <tr><td><code>dow</code>, <code>day_of_month</code></td><td>weekly and monthly seasonality</td></tr>
      <tr><td><code>category</code>, <code>subcategory</code></td><td>product type</td></tr>
      <tr><td><code>fc</code></td><td>which facility — footfall differs</td></tr>
      <tr><td><code>original_price</code></td><td>price <em>level</em>: a ₩30,000 steak and a ₩3,000 lettuce sell differently</td></tr>
      <tr><td><code>total_discount</code></td><td>the one price feature — <strong>overwritten to d_ref at inference, always</strong></td></tr>
    </tbody>
  </table></div>
  <figcaption>
    The two velocity features are the ones doing most of the work, and they are built
    deliberately: an <em class="term">anchor hour</em> is one priced within ±2.5pp of the
    category reference, so both rates measure <strong>how fast this SKU sells at roughly the
    reference price</strong> — not at whatever discount the legacy rule happened to apply.
    A velocity feature averaged over all discounts would smuggle the price effect back in
    through the side door, which is precisely what the overwritten price feature is there to
    prevent.
  </figcaption>
</figure>

<div class="callout">
  <h3>Why there is no “sales in the last hour” feature</h3>
  <p>
    It is the obvious feature to add, it would sharply improve offline accuracy, and it is
    excluded on purpose. Recent within-episode sales are a <strong>mediator of the
    episode's own price path</strong>: last hour's sales are partly the result of last
    hour's discount. Feed that back in and the model absorbs the price effect through the
    lag term, leaving the elasticity we are trying to learn understated — and it would do
    so invisibly, while every accuracy metric improved.
  </p>
  <p style="margin-top:14px">
    <code>hours_remaining</code> is excluded for a related reason: it is the agent's own
    state, and a model that knows how much clock is left will rediscover the legacy rule.
    The standing constraint is that <strong>one overwritten price feature is the auditable
    maximum</strong> — anything more and the level/slope split stops being checkable.
  </p>
</div>

<section class="prose">
  <h3 style="margin-top:34px">The unusual part</h3>
  <p>
    The model has a price feature and that feature is <strong>overwritten at inference
    time</strong>, always, to the reference discount. The model is therefore never asked
    what a different price would do; its price gradient is never queried. In the history it
    learns from, price was set by the clock — so a model free to use price would learn that
    correlation and report it back as price response. Blinding the model is what stops a
    confound from becoming a pricing decision.
  </p>
  <p>
    The demand model owns the <em class="term">level</em>. The learned posterior owns the
    <em class="term">slope</em>. Neither is permitted the other's job.
  </p>

  <h3 style="margin-top:34px">Demand is censored, and every comparison knows it</h3>
  <p>
    We never observe demand. We observe <em>sales</em>, which stop when the shelf is empty:
    “sold 2 of 2” means demand was <strong>at least</strong> 2. μ_ref is a statement about
    demand, so it can never be compared directly against a censored count.
  </p>
</section>

<div class="math">every prediction-vs-actual comparison uses  E[ min(D, q) ],  not μ

   D ~ NegBin(μ, r)      q = units on the shelf
   raw μ  ≥  E[min(D, q)]  always — so mixing the two reads systematically low</div>

<section class="prose">
  <p style="margin-top:16px">
    Fidelity, the calibration gate, and the level factors all run on that censored basis,
    and the DP applies the same <code>min(k, q)</code> inside its recursion.
  </p>

  <h3 style="margin-top:30px">What that means for training — where we made a trade</h3>
  <p>
    The model is trained on <code>units_sold</code>, and that label is itself censored. On an
    hour where the shelf emptied, the label records what fitted on the shelf, not what
    customers wanted. So μ_ref is fit toward something below true demand.
  </p>
  <p>
    The obvious response is to train only on hours that did not sell out. We tried it, and it
    is worse:
  </p>
</section>

<figure style="margin-top:18px">
  <div class="scroller"><table class="tl">
    <caption>Three ways to handle a censored label</caption>
    <thead><tr><th>Option</th><th>What happens</th><th>Verdict</th></tr></thead>
    <tbody>
      <tr>
        <td>Drop the censored hours</td>
        <td>Removes <strong>14% of rows</strong> — and not at random. Sell-outs concentrate in
            the later hours of a window, which is exactly when the selling actually happens.
            Training would be left with a reshaped population weighted toward the quiet part
            of the window</td>
        <td class="loss">rejected — selects on the outcome</td>
      </tr>
      <tr>
        <td>Fit a censored likelihood</td>
        <td>The principled fix: a sold-out hour enters as “demand ≥ q” rather than as a count.
            It is what we do everywhere else — fitting r, estimating the prior, every
            posterior update. But a <strong>censored label is not off-the-shelf under a
            Tweedie objective</strong>; it needs custom likelihood work and the validation to
            go with it</td>
        <td class="loss">right answer, out of scope for the MVP</td>
      </tr>
      <tr>
        <td>Keep the label, correct the level</td>
        <td>Accept a known downward bias in μ_ref, then measure and remove it in one place</td>
        <td class="good">chosen</td>
      </tr>
    </tbody>
  </table></div>
  <figcaption>
    The censoring signature is measurable rather than assumed. On the pre-calibration model
    the hourly bias runs at <strong>−0.09</strong> across uncensored hours and
    <strong>−0.21</strong> once censored hours are counted — the under-prediction more than
    doubles, which is what it looks like when the hours you cannot fully observe are the busy
    ones.
  </figcaption>
</figure>

<section class="prose">
  <p style="margin-top:26px">
    One thing makes that trade safe to take. <strong>Inventory and stockout indicators are
    deliberately not features.</strong> A model given them would learn to predict “the shelf
    was empty” — a statement about supply, not demand — and would carry it into every price
    the agent considers. Denied them, the model fits an average across stocked and stocked-out
    hours: biased low, but biased <em>smoothly</em>, which is precisely the kind of error a
    single level correction can remove.
  </p>
  <p>
    <strong>So the calibration step is not housekeeping — it is the other half of this
    decision.</strong> Having chosen to live with a censored label, we owe the system a
    measured correction for what that costs; and because the bias comes from censoring, the
    correction has to be solved on the censored basis to find it at all. The
    <strong>Calibration</strong> tab is that step.
  </p>
  <p>
    On the job it is given, the model does well: a measured sold ratio of
    <strong>0.9940</strong> across 2.07M rows, hourly error 0.4399 units. Accurate at the
    level, and never asked about the slope.
  </p>
</section>
"""

# ================================================================= 3 · calib
P_CALIB = f"""
<section class="prose" style="padding-top:34px">
  <div class="step-head"><h2>Correct the level. Never the slope.</h2></div>
  <p class="lede">
    A frozen model drifts against a moving world. Calibration is the one permitted
    correction: a single multiplicative factor per subcategory, nudging the demand level
    without touching anything else.
  </p>
</section>

{filecard("artifacts/calibration.json", "one multiplicative level factor per subcategory", "frozen, re-fit on schedule", "μ_ref, at inference")}

<section class="prose">
  <h3 style="margin-top:30px">Step 1 — what we are solving for</h3>
  <p>
    Not a ratio: a <em class="term">solve</em>. We look only at <strong>anchor rows</strong>
    — rows priced within 2.5pp of the reference discount, where the price-response term is
    exactly 1 and so cannot contaminate what we are measuring. Then we find the factor
    <code>f</code> that makes total predicted sales match total actual sales,
    <em>after</em> censoring:
  </p>
</section>

<div class="math">find f  such that   Σ E[ min( f · μ_i , q_i ) ]  =  Σ (units actually sold)
                    i ∈ anchor rows</div>

<section class="prose">
  <h3 style="margin-top:30px">Step 2 — the algorithm is bisection, and why it can be</h3>
  <p>
    Raise <code>f</code> and predicted sales can only go up — never down. That single
    property, monotonicity, is what lets us use the simplest search there is: guess halfway,
    see which side of the answer you are on, halve the interval, repeat. Forty halvings pin
    <code>f</code> to more decimal places than anyone needs, and unlike a gradient method it
    cannot diverge or land in a local trap.
  </p>
  <p>
    Here it is running on a deliberately round case: <strong>40 anchor rows, each with
    predicted demand μ = 1.0 and one unit on the shelf, and 28 units actually sold.</strong>
  </p>
</section>

<figure style="margin-top:18px">
  <div class="scroller"><table>
    <caption>Bisection for f · target = 28 units</caption>
    <thead><tr><th>Iteration</th><th class="raw">Try f</th><th class="raw">Σ E[min(f·μ, q)]</th><th>Verdict</th><th>New interval</th></tr></thead>
    <tbody>
      <tr><td>start</td><td>—</td><td>—</td><td>bracket the answer</td><td>[0.10, 10.00]</td></tr>
      <tr><td>1</td><td>5.050</td><td>32.83</td><td class="loss">too high</td><td>[0.10, 5.05]</td></tr>
      <tr><td>2</td><td>2.575</td><td>28.28</td><td class="loss">too high</td><td>[0.10, 2.58]</td></tr>
      <tr><td>3</td><td>1.337</td><td>22.48</td><td class="loss">too low</td><td>[1.34, 2.58]</td></tr>
      <tr><td>4</td><td>1.956</td><td>25.98</td><td class="loss">too low</td><td>[1.96, 2.58]</td></tr>
      <tr><td>5</td><td>2.266</td><td>27.23</td><td class="loss">too low</td><td>[2.27, 2.58]</td></tr>
      <tr><td>6</td><td>2.420</td><td>27.78</td><td class="loss">too low</td><td>[2.42, 2.58]</td></tr>
      <tr class="total"><td>… 40</td><td><strong>2.49</strong></td><td>28.00</td><td class="win">solved</td><td>—</td></tr>
    </tbody>
  </table></div>
  <figcaption>
    Computed with the system's own censoring code. Notice the interval collapsing by half
    each time, and notice how little movement there is between f = 2.27 and f = 2.58 — with
    one unit on the shelf, extra predicted demand mostly has nowhere to go. That flatness is
    the whole reason the next section exists.
  </figcaption>
</figure>

<section class="prose">
  <h3 style="margin-top:34px">Step 3 — why it is solved rather than divided</h3>
  <p>
    The naive way to get a correction is to divide: total sold ÷ total predicted. On the same
    40 rows that gives <code>28 ⁄ 40 = 0.70</code> — a factor below 1, meaning
    <em>“the model over-predicts, scale demand down.”</em>
  </p>
  <p>
    It is the wrong answer, and not by a little. At f = 1 the censored prediction is only
    <strong>19.7 units</strong>, because a shelf holding one unit cannot sell two however
    strong demand is. Against 28 actually sold, the model is <em>under</em>-predicting badly.
    The correct factor is <strong>2.49</strong>. The two methods disagree not about
    magnitude but about <strong>direction</strong>.
  </p>
</section>

<figure style="margin-top:18px">
  <div class="scroller"><table class="tl">
    <caption>The same 40 rows, two bases</caption>
    <thead><tr><th>Basis</th><th>What it computes</th><th>Factor</th><th>What it concludes</th></tr></thead>
    <tbody>
      <tr>
        <td>raw μ (divide)</td><td><code>28 ⁄ 40</code></td>
        <td class="loss">0.70</td><td class="loss">scale demand <em>down</em></td>
      </tr>
      <tr>
        <td>censored (solve)</td><td><code>Σ E[min(f·μ, q)] = 28</code></td>
        <td class="good">2.49</td><td class="good">scale demand <em>up</em></td>
      </tr>
    </tbody>
  </table></div>
  <figcaption>
    The same thing happened on production data, which is what the phrase “1.45 fits as 0.68”
    refers to: the correction that genuinely made censored predictions match reality was
    <strong>1.45</strong>, while dividing raw totals returned <strong>0.68</strong>. Because
    0.68 is below 1 and 1.45 is above it, a calibration built the naive way would have scaled
    demand in the opposite direction to the one the data called for — which is why it could
    never move the gate.
  </figcaption>
</figure>

<section class="prose">
  <h3 style="margin-top:34px">Step 4 — one factor per subcategory, pulled toward its parent</h3>
  <p>
    The solve above runs at <strong>subcategory</strong> level, because dispersion in demand
    level genuinely differs between, say, BERRY and CITRUS. But a subcategory with twelve
    anchor units sold has not earned its own aggressive correction. So each cell's raw factor
    is blended toward its parent category's factor, weighted by how much evidence it has:
  </p>
</section>

<div class="math">w  =  evidence ⁄ (evidence + 100)          evidence = anchor units sold
f  =  f_cell ^ w  ×  f_parent ^ (1 − w)    blended in log space, since factors multiply</div>

<figure style="margin-top:18px">
  <div class="scroller"><table>
    <caption>Three subcategories, same raw factor 1.62, parent category factor 1.20</caption>
    <thead><tr><th>Cell</th><th>Anchor units</th><th>Weight on itself</th><th>Blend</th><th>Factor applied</th></tr></thead>
    <tbody>
      <tr><td>thick</td><td>420</td><td>0.81</td><td>1.62<sup>0.81</sup> × 1.20<sup>0.19</sup></td><td class="good">1.529</td></tr>
      <tr><td>middling</td><td>100</td><td>0.50</td><td>1.62<sup>0.50</sup> × 1.20<sup>0.50</sup></td><td>1.394</td></tr>
      <tr><td>thin</td><td>12</td><td>0.11</td><td>1.62<sup>0.11</sup> × 1.20<sup>0.89</sup></td><td class="loss">1.239</td></tr>
    </tbody>
  </table></div>
  <figcaption>
    The pseudo-count is 100 anchor units, so a cell with 100 units of evidence trusts itself
    exactly half. A thick cell keeps almost all of its own signal; a thin one lands
    essentially on its parent. There is no threshold and therefore no cliff — a cell that
    gains its 101st unit does not suddenly change behaviour. Cells below 200 anchor rows are
    not fitted at all and simply inherit the parent.
  </figcaption>
</figure>

<section class="prose">
  <h3 style="margin-top:34px">What may never be corrected this way</h3>
  <p>
    Everything above fixes the <em>height</em> of the demand curve. Nothing above can fix
    its <em>tilt</em>. The diagnostic is the shape of the error as price moves away from the
    anchor:
  </p>
</section>

<figure style="margin-top:18px">
  <div class="cols two">
    <div>
      <svg class="diag" viewBox="0 0 340 210" role="img"
           aria-label="Level error: the ratio of actual to predicted sales sits at 1.45 and stays flat as the discount moves away from the reference, so one multiplicative factor corrects it.">
        <text class="cap" x="0" y="12">LEVEL ERROR — CORRECTABLE</text>
        <line class="ax" x1="42" y1="170" x2="330" y2="170"/>
        <line class="ax" x1="42" y1="26" x2="42" y2="170"/>
        <line class="ref" x1="42" y1="128" x2="330" y2="128"/>
        <text class="tick" x="36" y="132" text-anchor="end">1.0</text>
        <text class="tick" x="36" y="66" text-anchor="end">1.5</text>
        <line class="flat" x1="52" y1="76" x2="322" y2="72"/>
        <circle class="pt" cx="70" cy="75" r="3.5"/><circle class="pt" cx="130" cy="77" r="3.5"/>
        <circle class="pt" cx="190" cy="73" r="3.5"/><circle class="pt" cx="250" cy="74" r="3.5"/>
        <circle class="pt" cx="308" cy="72" r="3.5"/>
        <text class="tick" x="42" y="188">deeper</text>
        <text class="tick" x="186" y="188" text-anchor="middle">anchor</text>
        <text class="tick" x="330" y="188" text-anchor="end">shallower</text>
        <text class="note" x="52" y="100">off by the same amount everywhere</text>
      </svg>
    </div>
    <div>
      <svg class="diag" viewBox="0 0 340 210" role="img"
           aria-label="Slope error: the ratio is near 1 at the reference discount but degrades as the discount moves away from it, so no single multiplicative factor can correct it.">
        <text class="cap" x="0" y="12">SLOPE ERROR — NOT CORRECTABLE</text>
        <line class="ax" x1="42" y1="170" x2="330" y2="170"/>
        <line class="ax" x1="42" y1="26" x2="42" y2="170"/>
        <line class="ref" x1="42" y1="128" x2="330" y2="128"/>
        <text class="tick" x="36" y="132" text-anchor="end">1.0</text>
        <text class="tick" x="36" y="66" text-anchor="end">1.5</text>
        <path class="tilt" d="M52 46 C 120 92, 160 126, 190 128 C 232 131, 280 150, 322 158"/>
        <circle class="pt" cx="70" cy="55" r="3.5"/><circle class="pt" cx="130" cy="98" r="3.5"/>
        <circle class="pt" cx="190" cy="128" r="3.5"/><circle class="pt" cx="250" cy="143" r="3.5"/>
        <circle class="pt" cx="308" cy="156" r="3.5"/>
        <text class="tick" x="42" y="188">deeper</text>
        <text class="tick" x="186" y="188" text-anchor="middle">anchor</text>
        <text class="tick" x="330" y="188" text-anchor="end">shallower</text>
        <text class="note" x="196" y="112">right at the anchor,</text>
        <text class="note" x="196" y="124">wrong either side</text>
      </svg>
    </div>
  </div>
  <figcaption>
    Actual ÷ predicted sales, against distance from the reference discount. <strong>Left:</strong>
    a constant gap — one multiplication lands the whole line on 1.0, and that is what
    calibration does. <strong>Right:</strong> the model is right where it was fitted and
    wrong on both sides. Multiplying a wrong tilt by a constant leaves it wrong; it just
    moves where it crosses. That shape means the price-response term is off, and it is
    fixed by re-estimating the prior or by learning — never by scaling μ_ref. The two
    shapes are drawn to show the diagnostic, not measured values.
  </figcaption>
</figure>

<figure style="margin-top:30px">
  <div class="scroller"><table class="tl">
    <caption>Where the gate stands</caption>
    <thead><tr><th>Check</th><th>Value</th><th>Verdict</th></tr></thead>
    <tbody>
      <tr><td>level at the reference price</td><td>1.0389</td><td class="good">PASS — band [0.90, 1.10]</td></tr>
    </tbody>
  </table></div>
  <figcaption>
    The band is roughly two standard deviations of measured week-to-week demand volatility
    — wide enough not to fire on ordinary noise, tight enough to catch a model that has
    genuinely gone stale. The factors are fit on train and calibration weeks and graded on
    test, so the gate never grades its own fit.
  </figcaption>
</figure>
"""

# =================================================================== 4 · var
def _bar_row(label, vals, hi):
    """One sales-count row across the three dispersions."""
    cells = "".join(
        f'<div class="track"><div class="bar {cls}" style="width:{v / hi * 100:.1f}%"></div>'
        f'<b>{v:.1f}%</b></div>'
        for v, cls in zip(vals, ("lumpy", "nb", "po")))
    return (f'<div class="row"><span>{label}</span>'
            f'<div class="bar-pair">{cells}</div></div>')


_PMF = [("sells 0", [71.6, 64.6, 57.6]), ("sells 1", [15.4, 22.5, 31.4]),
        ("sells 2", [6.4, 8.2, 9.0]), ("sells 3", [3.1, 3.0, 1.8]),
        ("sells 4+", [3.5, 1.7, 0.2])]

P_VAR = f"""
<section class="prose" style="padding-top:34px">
  <div class="step-head"><h2>Two frozen numbers that decide how fast we may believe</h2></div>
  <p class="lede">
    These artifacts never touch a price. They govern something subtler and more dangerous
    to get wrong: <strong>how much a piece of evidence is worth</strong>. Set them badly and
    the system learns confidently from data that does not support the confidence.
  </p>
</section>

{filecard("artifacts/r_lookup.json", "dispersion r, by subcategory with a fallback chain", "frozen", "every probability the DP uses")}

<section class="prose">
  <h3 style="margin-top:30px">What r is</h3>
  <p>
    The obvious model for counting sales is Poisson, and it is wrong here — real perishable
    demand has more dead hours <em>and</em> more sudden baskets than Poisson allows. We use
    a negative binomial, whose spread is governed by a single number:
  </p>
</section>
<div class="math">Var[D]  =  μ  +  μ² ⁄ r         small r → lumpy    ·    large r → Poisson</div>

<div class="callout">
  <h3>The same average demand, three different worlds</h3>
  <div class="key" style="margin-top:2px">
    <span><i class="lumpy"></i>r = 0.35 lumpy</span>
    <span><i class="nb"></i>r = 0.919 measured</span>
    <span><i class="po"></i>r = 20 near-Poisson</span>
  </div>
  <div class="bars">
    {"".join(_bar_row(l, v, 75) for l, v in _PMF)}
  </div>
  <p style="margin-top:16px">
    Every column has mean <strong>0.56 units</strong>. Only the spread differs. Read the top
    row: as demand gets lumpier the chance of an hour selling <em>nothing at all</em> climbs
    from 58% to 72% — and the tail gets fatter at the same time. An agent that assumed
    Poisson would be wrong in both directions simultaneously, and confidently.
  </p>
</div>

<section class="prose">
  <h3 style="margin-top:34px">What it costs to get it wrong</h3>
  <p>
    Not an abstraction. Take the worked episode from the decision tab and swap only the
    demand spread, holding the average fixed:
  </p>
</section>

<figure style="margin-top:18px">
  <div class="scroller"><table class="tl">
    <caption>Same episode, same average demand, different assumed spread</caption>
    <thead><tr><th>Assumption</th><th>What the agent thinks the episode is worth</th><th>Error</th></tr></thead>
    <tbody>
      <tr><td>demand is “well behaved” (Poisson)</td><td class="loss">−₩4,254</td><td class="loss">13% too optimistic</td></tr>
      <tr><td>measured spread, r = 0.919</td><td class="good">−₩4,881</td><td class="good">the honest answer</td></tr>
    </tbody>
  </table></div>
  <figcaption>
    The optimism is not random — it runs in one direction, every hour. Assume demand is
    tidier than it is and the agent systematically overestimates the chance that holding
    price will still clear the shelf, so it holds when it should not. A variance error
    becomes a pricing error.
  </figcaption>
</figure>

<section class="prose">
  <h3 style="margin-top:34px">How r is fitted</h3>
  <p>
    By maximum likelihood — we ask which value of <code>r</code> makes the hours we actually
    observed most probable. The only subtlety is that a sold-out hour is not a count, it is a
    lower bound, so it enters through a different term:
  </p>
</section>

<div class="math">choose r to maximise:

    Σ  log P( D = k  | μ, r )      hours that did not sell out   — an exact count
  + Σ  log P( D ≥ q  | μ, r )      hours that did sell out       — “at least q”

  searched over r ∈ [0.05, 50] by bounded scalar optimisation on log r</div>

<section class="prose">
  <p style="margin-top:16px">
    Searching on <code>log r</code> rather than <code>r</code> matters: the difference between
    r = 0.3 and r = 0.6 is enormous for the shape of the distribution, while the difference
    between r = 30 and r = 60 is nearly nothing. Log space makes the search spend its effort
    where the answer changes. It is a one-dimensional bounded search, so it is fast and cannot
    wander off.
  </p>
  <p>
    Fitted per <strong>subcategory</strong> where there are at least 200 rows to support it,
    otherwise falling back a level: <strong>subcategory → category → global</strong>.
  </p>

  <h3 style="margin-top:34px">Why r is not shrunk, when the calibration factor is</h3>
  <p>
    A fair question, since the two problems look alike. Two reasons they are handled
    differently.
  </p>
  <p>
    <strong>A level is not a shape.</strong> The calibration factor is one multiplicative
    number, so “halfway between this cell and its parent” has an obvious meaning — multiply
    by the geometric mean. <code>r</code> is a shape parameter that bends an entire
    distribution; a blend of two r values is not the distribution that sits halfway between
    the two, and the blended number would not be the maximum-likelihood answer to any
    question we asked.
  </p>
  <p>
    <strong>The danger is one-sided.</strong> Being wrong about r is not symmetric: a
    too-high r means near-Poisson, which makes the agent overconfident and every learning
    update too bold. Shrinking a thin, genuinely lumpy cell toward a tidier parent would move
    it in exactly that direction — the shrinkage would be doing harm precisely where the
    evidence is weakest.
  </p>
</section>

<figure style="margin-top:18px">
  <div class="scroller"><table class="tl">
    <caption>So the safeguards are deliberately asymmetric</caption>
    <thead><tr><th>Situation</th><th>What happens</th><th>Why that direction</th></tr></thead>
    <tbody>
      <tr><td>thin cell (&lt; 200 rows)</td><td>fall back to the parent outright — no blending</td><td>an inherited estimate from real data beats an invented compromise</td></tr>
      <tr><td>fitted r comes back <em>high</em></td><td>clamped at the 90th percentile of converged values</td><td>the optimistic direction: we refuse to be talked into near-Poisson by a thin cell</td></tr>
      <tr><td>fitted r comes back <em>low</em></td><td>kept exactly as fitted, uncapped</td><td>the safe direction: a lumpy estimate only ever makes the agent more careful</td></tr>
    </tbody>
  </table></div>
  <figcaption>
    The rule of thumb behind all three rows: when unsure, be wrong toward lumpier. That error
    costs a little speed. The opposite error costs correctness.
  </figcaption>
</figure>

{filecard("artifacts/rho.json", "ρ, mean forced hours, and the resulting evidence deflator", "frozen", "every posterior update")}

<section class="prose">
  <h3 style="margin-top:30px">ρ and deff, without the statistics</h3>
  <p>
    Suppose you want to know whether a restaurant is any good. Ten reviews from ten different
    diners tell you much more than ten reviews from one diner who ate there ten times — the
    second set mostly tells you about that one person.
  </p>
  <p>
    Hours inside one clearance episode are the repeat diner. They share one inventory pool,
    one afternoon's weather, one store's footfall — if it is a quiet Tuesday, it is quiet for
    all of them. <strong>ρ measures how much of the leftover surprise belongs to the episode
    rather than to the hour.</strong>
  </p>

  <h3 style="margin-top:30px">How ρ is computed</h3>
  <p>
    Take every hour the model predicted, and record what it got wrong:
  </p>
</section>

<div class="math">residual        =  units actually sold  −  μ predicted for that hour

ρ  =  variance of the EPISODE-AVERAGE residuals
      ────────────────────────────────────────    over episodes of ≥ 3 hours
              variance of ALL residuals

     =  0.3103        clipped to [0, 0.95]</div>

<section class="prose">
  <p style="margin-top:16px">
    The intuition is direct. If every hour's miss were independent noise, episode averages
    would wash out to nearly nothing and the ratio would sit near zero. If a whole episode
    were quiet together, the episode averages would carry most of the spread and the ratio
    would approach one. At <strong>0.31</strong>, roughly a third of what the model gets wrong
    is a property of the episode, not of the hour — and that third does not average away no
    matter how many hours of the same episode we collect.
  </p>
  <p>
    It is computed against the model's own residuals, not from raw sales, which is the only
    way to isolate correlation the model has <em>not</em> already explained through hour of
    day and the rest.
  </p>

  <h3 style="margin-top:30px">From ρ to deff</h3>
  <p>
    The conversion is a standard result from survey sampling — the
    <em class="term">design effect</em> for clustered observations. If you sample in clusters
    of size <code>m</code> with within-cluster correlation ρ, the effective sample size is the
    real one divided by:
  </p>
</section>

<div class="math">deff  =  1 + (m − 1) × ρ            m = the cluster size

      =  1 + (8.563 − 1) × 0.3103
      =  3.347</div>

<section class="prose">
  <p style="margin-top:16px">
    <strong>Why m = 8.563.</strong> The cluster is one episode, and its size is how many
    priced hours it contributes. But not every episode contributes evidence: we take the mean
    length of episodes <em>in which the price actually moved at least once</em> — the ones
    capable of saying anything about price response. Averaging over all episodes, including
    the ones that sat at one price throughout, would understate the cluster size and so
    understate the tax.
  </p>
  <p>
    Read the formula at its two extremes and it explains itself. With ρ = 0 the hours are
    independent, deff = 1, and nothing is deflated. With ρ = 1 the whole episode is one
    observation repeated, deff = m, and eight hours count as one. We are at 3.347: eight and a
    half hours count as about two and a half.
  </p>
</section>

<div class="callout">
  <h3>What that means in practice</h3>
  <p>
    A pilot day producing <strong>100 episodes</strong> generates about 856 priced hours.
    The system does not count them as 856 observations. It counts them as
    <strong>256</strong> — and the belief moves accordingly.
  </p>
  <p style="margin-top:14px">
    That deflation is the difference between the loop below and the loop it would otherwise
    run. An update commits when accumulated <em>effective</em> information reaches 12.0, and
    a shadow measurement put one episode at 0.0087 of it:
  </p>
</div>

<div class="math">episodes per committed step  =  12.0 ⁄ 0.0087  ≈  1,381</div>

<section class="prose">
  <p style="margin-top:16px">
    Without the deflation that number would read about 400, and the system would announce
    convergence three and a half times too early — on evidence that was mostly one quiet
    Tuesday, repeated. ρ is deliberately one global scalar rather than one per category: a
    noisy per-cell estimate would deflate evidence unevenly, penalising exactly the cells
    with the least data. Start-up refuses to run if the configured value has drifted from
    the frozen artifact.
  </p>
</section>
"""

# ================================================================= 5 · prior
P_PRIOR = f"""
<section class="prose" style="padding-top:34px">
  <div class="step-head"><h2>The one number history could not give us</h2></div>
  <p class="lede">
    Price response is the quantity the whole system turns on, and it is the one thing we do
    not know. We tried to estimate it from history, in the most defensible form we could
    construct. <strong>The attempt failed, and understanding exactly how it failed is the
    argument for everything else on this page.</strong>
  </p>
</section>

{filecard("artifacts/prior.json", "the cold-start price response: −1.0 ± 0.6, every category", "frozen cold start", "the DP, until production learns better")}

<section class="prose">
  <h3 style="margin-top:30px">First: what ε actually does</h3>
  <p>
    One exponent moves the demand model's answer from the reference price to any other
    price. It is the entire price mechanism in the system:
  </p>
</section>

<div class="math">μ(d)  =  μ_ref · ( (1 − d) ⁄ (1 − d_ref) ) ^ ε        ε constrained negative

at d = 0     μ = 0.80 · (1.00 ⁄ 0.70) ^ −1  =  0.56     full price
at d = 0.30  μ = 0.80                                   the reference itself
at d = 0.60  μ = 0.80 · (0.40 ⁄ 0.70) ^ −1  =  1.40     at the cost floor</div>

<section class="prose">
  <p style="margin-top:16px">
    The ML model is consulted once per hour and that single number is swung across the whole
    price ladder by this exponent. Get ε wrong and every price the agent considers is
    mispriced — not by a little, and in a direction that compounds over a window.
  </p>

  <h3 style="margin-top:34px">What we tried</h3>
  <p>
    The procedure — the <em class="term">bracket</em> — was built around the two confounds
    rather than in spite of them. If we cannot get one unbiased estimate, we can try to get
    <strong>two biased ones that we know lean opposite ways</strong>, and let them fence the
    answer in.
  </p>
</section>

<section class="prose">
  <p>
    Both estimates come from the same likelihood — we score every candidate ε on a grid across
    the whole feasible range and keep the one that makes the observed sales most probable:
  </p>
</section>

<div class="math">for each candidate ε on a grid across [−4.00, −0.05]:

    μ_i(ε)  =  μ_ref,i · exp( ε · L_i )         L_i = log( (1 − d_i) ⁄ (1 − d_ref) )

    log L(ε)  =  Σ  log P( D = k_i | μ_i(ε), r_i )     hours that did not sell out
              +  Σ  log P( D ≥ q_i | μ_i(ε), r_i )     hours that did

    ε̂  =  argmax  log L(ε)</div>

<section class="prose">
  <p style="margin-top:16px">
    The two versions differ in exactly one place, and that difference is the whole experiment:
  </p>
</section>

<figure style="margin-top:18px">
  <div class="scroller"><table class="tl">
    <caption>Two estimates per category, one likelihood, one difference</caption>
    <thead><tr><th>Estimate</th><th>What it does to the likelihood</th><th>Which way it is wrong</th></tr></thead>
    <tbody>
      <tr>
        <td><code>ε_naive</code></td>
        <td>nothing — μ_i(ε) is used exactly as written above</td>
        <td>every hour-of-day effect the model did not already capture is left for ε to
            explain, and since deep discounts happen in busy hours, ε absorbs the evening
            lift → <strong>too elastic</strong></td>
      </tr>
      <tr>
        <td><code>ε_controlled</code></td>
        <td>multiplies μ_i(ε) by an hour-of-day factor <code>h(hour)</code>, re-estimated at
            every candidate ε by moment matching on uncensored rows:
            <code>h = Σ sold ⁄ Σ μ</code> within each hour</td>
        <td>the hour factors soak up the confound — but they also soak up most of the genuine
            price variation, which under a clock-driven policy is nearly collinear with hour
            → <strong>biased toward zero</strong></td>
      </tr>
    </tbody>
  </table></div>
  <figcaption>
    The controlled version is the honest attempt, and its weakness is structural rather than
    fixable: when price is a function of the hour, controlling for the hour removes the price
    signal along with the confound. Profiling the hour factors <em>at each candidate ε</em>
    rather than fixing them once is what makes it a fair fight — the control adapts to
    whatever ε is being tested.
  </figcaption>
</figure>

<section class="prose">
  <p style="margin-top:26px">
    If both are honest, the truth lies between them. So the prior becomes the midpoint and
    the gap becomes the uncertainty:
  </p>
</section>

<div class="math">prior mean  =  ( ε_naive + ε_controlled ) ⁄ 2
prior std   =  max( | ε_naive − ε_controlled | ⁄ 2 , floor )

accept only if:   ε_naive  ≤  ε_controlled  <  0        ← orientation holds
                  neither estimate sits on a search bound</div>

<section class="prose">
  <p style="margin-top:16px">
    The orientation check is the clever part and the one that failed. We know in advance which
    way each estimate must lean — naive more elastic, controlled less. If that ordering comes
    back reversed, the two biases are not behaving as theory says they must, which means
    something else is driving the fit and neither number can be trusted. It is a test the
    procedure can fail on its own terms, without anyone judging whether a number looks
    plausible.
  </p>
</section>

<section class="prose">
  <p style="margin-top:26px">
    Both estimates run on <strong>entry-hour rows only</strong> — the first priced hour of
    each episode, compared across episodes rather than within one. That restriction is the
    survivorship confound made operational: within an episode, a row at a deep discount
    exists <em>because</em> the earlier hours did not sell, so those rows are drawn from the
    slow tail by construction. Entry rows carry genuine cross-episode variation (different
    start hours, different cost floors) without that selection.
  </p>
  <p>
    Both also search the full support, <code>[−4.0, −0.05]</code>. An earlier run searched a
    narrower range and pinned five categories at its edge, which looks like agreement and is
    actually a cage.
  </p>

  <h3 style="margin-top:34px">What came back</h3>
</section>

<figure style="margin-top:16px">
  <div class="scroller"><table class="tl">
    <thead><tr><th>What we found</th><th>What it means</th></tr></thead>
    <tbody>
      <tr>
        <td><span class="num">−0.10</span> vs <span class="num">−3.05</span></td>
        <td>one category's two estimates — <strong>30× apart</strong>. The bracket is not a
            bracket; it spans nearly the whole feasible range, and its midpoint is an
            artifact of where two biases happened to land</td>
      </tr>
      <tr>
        <td>pinned at the search bound</td>
        <td>another category's estimate sat on the edge of the grid. A boundary solution is
            not an estimate — it is an optimiser reporting its cage, and it is rejected
            outright rather than read as “very elastic”</td>
      </tr>
    </tbody>
  </table></div>
  <figcaption>
    Acceptance was never about whether the number looked plausible. It rested on orientation
    (do the two estimates lean the way theory says they must?) and sufficiency (is there
    enough identifying variation to believe either?). <strong>No category cleared both.</strong>
    The instability is the fingerprint: the fitted optimum is set by whatever variation
    dominates the data — the clock, survivorship, weekly shocks — not by customer price
    response.
  </figcaption>
</figure>

<div class="callout">
  <h3>Why a fitted number would have been worse than an honest guess</h3>
  <p>
    A fitted ε makes the offline replay look excellent, because it fits the residual by
    construction. What it teaches the agent is the confound: <em>hold price, burn scrap</em>
    — the very behaviour the legacy rule already produces. The dashboard would smile while
    the system failed, and we would have no instrument capable of noticing.
  </p>
</div>

<section class="prose">
  <h3 style="margin-top:34px">Why −1.0, specifically</h3>
  <p>
    Having rejected the estimates, we still have to start somewhere. −1.0 is chosen to be
    <strong>conservative in the direction that matters</strong>: it is a modest belief in
    price response, so the agent discounts sparingly and holds price where it is unsure.
    A larger guess — say −2.5 — would have the agent marking down aggressively on day one on
    the strength of a number we just finished arguing we do not know.
  </p>
  <p>
    That asymmetry is deliberate because <strong>clearance is not the problem</strong>.
    Clearance already runs around <strong>91%</strong>; the business is not losing money for
    want of selling the stock. It is losing money on the terms of the sale. The objective is
    Inventory Loss, and the cheapest mistake to recover from is holding price a little too
    long. Discounting too hard gives away margin that no later decision can win back.
  </p>
</section>

"""


# ================================================================ 6 · learning
P_LEARN = f"""
<section class="prose" style="padding-top:34px">
  <div class="step-head"><h2>The one artifact that moves</h2></div>
  <p class="lede">
    Everything on the other tabs is written once and then only read. This one is written by
    production, and it holds the single quantity history could not give us. How fast it moves
    is, in a real sense, the product.
  </p>
</section>

{filecard("artifacts/posterior.json", "the current belief about price response, per learning cell", "the only one that moves", "the DP, every hour", moves=True)}

<section class="prose">
  <h3 style="margin-top:34px">The vocabulary, first</h3>
  <p>
    Four terms do most of the work on this tab. They are worth fixing before the arithmetic.
  </p>
</section>

<figure style="margin-top:14px">
  <div class="scroller"><table class="tl">
    <thead><tr><th>Term</th><th>What it means here</th></tr></thead>
    <tbody>
      <tr><td><strong>rung</strong></td>
          <td>one step on the discount ladder — 2.5 percentage points. “Two rungs down” from
              ₩10,000 is 5% off, ₩9,500. The ladder is the action set from the decision tab</td></tr>
      <tr><td><strong>cost of a rung</strong></td>
          <td>expected won of Inventory Loss given up by pricing there instead of at the best
              rung. Not a forecast — a subtraction between two numbers the agent already
              computed</td></tr>
      <tr><td><strong>τ (tau)</strong></td>
          <td>the budget line, in won. A rung is <em>affordable</em> if its cost is at or
              below τ</td></tr>
      <tr><td><strong>effective information</strong></td>
          <td>how much a batch of outcomes actually tells us about price response, after
              discounting the fact that hours from one episode partly repeat each other.
              Raw information ÷ deff, and an update commits only when it reaches 12.0</td></tr>
    </tbody>
  </table></div>
</figure>

<section class="prose">
  <h3 style="margin-top:34px">Choosing where to experiment</h3>
  <p>
    The agent has already scored every rung. So the price of deviating from the best one is
    not estimated — it is a subtraction it has in hand:
  </p>
</section>

<div class="math">cost(p)     =  Q(p*) − Q(p)                  ← won of expected IL given up
affordable  =  {{ p ≠ p*  :  cost(p) ≤ τ }}
applied     =  UNIFORM draw from affordable   ← never weighted, never targeted</div>

<section class="prose">
  <p style="margin-top:16px">
    Uniformity is the mechanism, not a simplification. Any weighting by belief or state
    reintroduces the correlation between price and circumstance that made the history
    unusable in the first place. A uniform draw inside a budget band is what turns an outcome
    into causal evidence.
  </p>

  <h3 style="margin-top:30px">The subtraction, in full</h3>
  <p>
    Here is the worked episode from the decision tab — three units, four hours left. These are
    the solver's own scores for the top four rungs:
  </p>
</section>

<figure style="margin-top:18px">
  <div class="scroller"><table>
    <caption>Cost of each rung · 3 units · 4 hours left · τ = ₩447.78</caption>
    <thead>
      <tr><th>Rung</th><th>Price</th><th>Q — what the episode is worth if priced here</th><th>cost = Q(best) − Q</th><th>≤ τ ?</th></tr>
    </thead>
    <tbody>
      <tr><td>0 · the best</td><td>₩10,000</td><td>−4,881</td><td>₩0</td><td>—</td></tr>
      <tr><td>1 rung down</td><td>₩9,750</td><td>−5,218</td><td class="good">₩338</td><td class="good">yes — affordable</td></tr>
      <tr><td>2 rungs down</td><td>₩9,500</td><td>−5,568</td><td class="loss">₩687</td><td class="loss">no</td></tr>
      <tr><td>3 rungs down</td><td>₩9,250</td><td>−5,930</td><td class="loss">₩1,049</td><td class="loss">no</td></tr>
    </tbody>
  </table></div>
  <figcaption>
    −5,218 − (−4,881) = <strong>−337</strong>, so pricing one rung down costs about ₩338 of
    expected loss. That is inside the ₩447.78 budget, so it is a legal experiment. Two rungs
    down costs ₩687 and is not. The affordable set here has exactly
    <strong>one</strong> member, and the agent draws uniformly from it.
  </figcaption>
</figure>

<section class="prose">
  <p style="margin-top:26px">
    Now the same threshold in two other states, and the point of the whole design appears:
  </p>
</section>

<figure style="margin-top:16px">
  <div class="scroller"><table>
    <caption>One threshold, three states</caption>
    <thead>
      <tr><th>State</th><th>1 rung</th><th>2 rungs</th><th>3 rungs</th><th>4 rungs</th><th>Affordable</th></tr>
    </thead>
    <tbody>
      <tr><td>3 units · 4 hrs left</td>
          <td class="good">₩338</td><td class="loss">₩687</td><td class="loss">₩1,049</td><td class="loss">₩1,424</td>
          <td><strong>1 rung</strong></td></tr>
      <tr><td>1 unit · 4 hrs left</td>
          <td class="good">₩184</td><td class="good">₩370</td><td class="loss">₩560</td><td class="loss">₩753</td>
          <td><strong>2 rungs</strong></td></tr>
      <tr><td>1 unit · 1 hr left</td>
          <td class="good">₩67</td><td class="good">₩136</td><td class="good">₩208</td><td class="good">₩282</td>
          <td><strong>6 rungs</strong></td></tr>
    </tbody>
  </table></div>
  <figcaption>
    With three units and four hours the agent has a great deal riding on the price and the
    value curve is steep — one rung is all the budget buys. With one unit and one hour left,
    the episode is nearly decided whatever we do, deviating costs almost nothing, and six
    rungs open up. <strong>Exploration concentrates itself where the agent is closest to
    indifferent — which is exactly where the answer is least known.</strong> No
    state-dependent rule produces that. The budget does, on its own.
  </figcaption>
</figure>

<section class="prose">
  <h3 style="margin-top:34px">Why there is nothing cleverer to do</h3>
  <p>
    A natural question follows: if some experiments are more informative than others, should
    we not spend the budget on those? The answer is that they are not.
  </p>
  <p>
    How much an hour tells us about ε is a standard quantity — how sharply the predicted
    demand changes when ε changes. Our demand equation makes that derivative easy:
  </p>
</section>

<div class="math">μ(ε)  =  μ_ref · e^(ε·L)          L = log( (1 − d) ⁄ (1 − d_ref) ),  the log price ratio

dμ ⁄ dε  =  μ · L                  a bigger price move, and a busier hour, both move μ more

information  ≈  (dμ ⁄ dε)² ⁄ Var[D]  ≈  (μL)² ⁄ μ  =  μ · L²</div>

<section class="prose">
  <p style="margin-top:16px">
    So an experiment's information is <strong>demand × (how far the price moved, in logs)
    squared</strong>. Both halves are intuitive: a busier hour gives a bigger sample, and a
    bigger price move gives a longer lever.
  </p>
  <p>
    The catch is that the <em>cost</em> of the perturbation scales with the same product. More
    demand means more units sold at the discounted price; a longer lever means a deeper
    discount on each. So <strong>information per won is roughly constant</strong> — a won
    spent on a slow SKU at a deep discount buys about what a won spent on a busy SKU at a
    shallow one buys. There is no informative region to steer toward, because every region is
    equally informative per unit of budget. There is only a budget to respect.
  </p>

  <h3 style="margin-top:34px">Where τ comes from — and what happens to it at launch</h3>
  <p>
    τ is derived, never chosen. We search for the threshold whose implied daily spend equals
    the approved budget:
  </p>
</section>

<div class="math">τ₀      :  bisect until  implied daily spend  =  1% of daily IL
τ_next  =  τ × clip( budget ⁄ realised spend, 0.5, 2.0 )      recomputed daily
budget  =  1% of IL × clip( posterior std ⁄ 0.40, 0.25, 1 )   shrinks as belief tightens</div>

<section class="prose">
  <p style="margin-top:16px">
    The <strong>₩447.78</strong> quoted today was derived on the replay population — a sample
    of prepared history, not the launch book. It is a starting point, not a production
    constant, and two things fix that. It is <strong>re-derived before launch</strong> from a
    backtest on the launch population that has itself passed the calibration gate; and then it
    <strong>corrects itself daily</strong> against realised spend. The clip is the safety —
    at most 2× a day, so one anomalous day cannot run away with the budget, and a starting
    value wrong by a factor of four is gone within two days.
  </p>

  <h3 style="margin-top:34px">How the belief updates</h3>
  <p>
    Not a formula applied to a summary. The full posterior is recomputed on a grid over the
    whole feasible range, every time:
  </p>
</section>

<div class="math">for each ε on a grid across [−4.00, −0.05]:

    log L(ε)  =  Σ  log P( D = k | μ_ref·e^(ε·L), r )     hours that did not sell out
              +  Σ  log P( D ≥ q | μ_ref·e^(ε·L), r )     hours that did

    log posterior(ε)  =  log L(ε)  −  ½ ( (ε − mean) ⁄ std )²   ← the belief we started with

    normalise across the grid, then take its mean and standard deviation as the new belief</div>

<section class="prose">
  <p style="margin-top:16px">
    Two details are doing real work. Stockout hours enter through <strong>P(D ≥ q)</strong>,
    not as counts — “sold 2 of 2” means demand was at least 2, and treating it as exactly 2
    would understate price response precisely where stockouts concentrate, at deep discounts.
    And zero-sale hours are kept, because they carry the shallow-discount signal that nothing
    else does.
  </p>
</section>

<figure style="margin-top:20px">
  <div class="scroller"><table class="tl">
    <caption>What the raw update is not allowed to do</caption>
    <thead><tr><th>Guard</th><th>Limit</th><th>Why</th></tr></thead>
    <tbody>
      <tr><td>Evidence bar</td><td>effective information ≥ 12.0</td><td>a short batch is not consumed at all — it waits, rather than being spent for a small move</td></tr>
      <tr><td>Step cap</td><td>mean moves ≤ 0.15</td><td>no single batch can swing the pricing policy</td></tr>
      <tr><td>Confidence cap</td><td>std shrinks ≤ 25%</td><td>no single batch can manufacture certainty</td></tr>
      <tr><td>Human gate</td><td>one approved update per day</td><td>and the apply command refuses while event-quality gates fail</td></tr>
      <tr><td>Ledger</td><td>one atomic commit</td><td>the revised belief and the outcome IDs it consumed are written together, so a crash cannot double-count and evidence is never spent twice</td></tr>
    </tbody>
  </table></div>
  <figcaption>
    Every guard makes learning slower. That is the intended direction: the failure that
    matters here is not learning late, it is becoming confident early and wrong.
  </figcaption>
</figure>

<section class="prose">
  <h3 style="margin-top:34px">So how long does it take?</h3>
  <p>
    It is tempting to count steps to the point where the agent would start discounting — the
    belief starts at 1.0, the switch sits somewhere above 2, the cap is 0.15 a step, so ten
    updates. <strong>The arithmetic is right and the question is wrong.</strong> Where true ε
    sits is exactly what we do not know; that is the premise of the project. It may be high,
    in which case the agent will start deepening. It may be low, in which case the agent holds
    price and <em>that is the correct answer</em>, not a failure to converge. Planning a
    duration around reaching the switch would be assuming the conclusion.
  </p>
</section>

<figure style="margin-top:20px">
  <div class="scroller"><table class="tl">
    <caption>What we can commit to in advance, and what we cannot</caption>
    <thead><tr><th>Question</th><th>Decided by</th><th>Answer</th></tr></thead>
    <tbody>
      <tr>
        <td>How long does the experiment run?</td>
        <td>statistical power, fixed before launch</td>
        <td>Measured standard error <strong>0.002915</strong> gives <strong>6.75%</strong>
            detectable at two weeks. Six weeks reaches only 5.74% — the curve is nearly flat,
            because the variance is between-unit and the same units recur weekly. More weeks
            buy very little; more SKU × FC units would</td>
      </tr>
      <tr>
        <td>When will the belief move enough to change prices?</td>
        <td>the customers</td>
        <td>Unknown, and not ours to schedule. We report the gap every run so it is watched
            rather than assumed</td>
      </tr>
      <tr>
        <td>Is the loop alive at all?</td>
        <td>instrumentation</td>
        <td>Posterior width changes only when an update commits, so “width unchanged for 21
            days” is exactly “no learning for 21 days” — one cheap alert that catches every
            cause at once. The age of the oldest unconsumed outcome is reported daily</td>
      </tr>
    </tbody>
  </table></div>
  <figcaption>
    The honest framing for a launch decision: the experiment's length is a power question we
    have already answered, and the learning loop's progress is monitored rather than promised.
  </figcaption>
</figure>
"""

PANELS = {"map": P_MAP, "data": P_DATA, "model": P_MODEL, "calib": P_CALIB,
          "var": P_VAR, "prior": P_PRIOR, "learn": P_LEARN}


# ================================================================== 7 · replay
P_REPLAY = f"""
<section class="prose" style="padding-top:34px">
  <div class="step-head"><h2>Replaying five months, to see what the agent would have done</h2></div>
  <p class="lede">
    Before any real price moves, we re-run history: take each episode as it actually
    happened — the same SKU, the same stock, the same window — and step through it hour by
    hour, asking the agent what it would have priced. Then compare.
  </p>
  <p>
    The comparison is where almost everyone goes wrong, so it is worth being slow about it.
    Replay produces <strong>three different quantities</strong>, and only one pairing of them
    is a statement about the policy.
  </p>
</section>

<figure style="margin-top:20px">
  <div class="scroller"><table class="tl">
    <caption>Three numbers replay produces, and what each one is</caption>
    <thead><tr><th>Quantity</th><th>What it is</th><th>What it can be used for</th></tr></thead>
    <tbody>
      <tr>
        <td><strong>observed</strong><br><code>actual_*</code></td>
        <td>What really happened in the world, under the legacy rule. Real sales, real scrap,
            real money</td>
        <td class="loss">Fidelity only — is our demand model a decent description of reality?
            <strong>Never a policy statement</strong></td>
      </tr>
      <tr>
        <td><strong>legacy, simulated</strong><br><code>legacy_model_*</code></td>
        <td>The legacy rule's prices, replayed through our demand model instead of the world</td>
        <td class="good">The control arm</td>
      </tr>
      <tr>
        <td><strong>agent, simulated</strong><br><code>dp_*</code></td>
        <td>The agent's prices, replayed through the <em>same</em> demand model</td>
        <td class="good">The treatment arm</td>
      </tr>
    </tbody>
  </table></div>
  <figcaption>
    The tempting comparison is agent-simulated against observed — our new system against what
    actually happened. It is wrong, and badly so: it charges <em>every</em> error in the
    demand model to the agent's account. A model that over-predicts demand would make the
    agent look brilliant against reality while telling us nothing about its pricing.
  </figcaption>
</figure>

<div class="callout">
  <h3>The only honest comparison is like-for-like</h3>
  <div class="math" style="margin-top:0">legacy-under-model IL   vs   agent-under-model IL

same μ_ref · same r · same ε  →  model bias hits both arms and cancels</div>
  <p style="margin-top:16px">
    Both arms are simulated by the identical demand generator. Whatever our model gets wrong,
    it gets wrong for the legacy rule too, so the difference between the arms is attributable
    to the <em>decisions</em> and to nothing else. That difference is the only number in this
    tab that says anything about the policy.
  </p>
</div>

<section class="prose">
  <h3 style="margin-top:34px">Say this part out loud, before someone finds it</h3>
  <p>
    A markdown system that reduces loss by <em>discounting less</em> is not what most people
    expect. It is what the numbers say. The agent does not clear more stock — it clears
    slightly <strong>less</strong>, and scraps slightly <strong>more</strong>.
  </p>

  <h3 style="margin-top:34px">All three, side by side</h3>
  <p>
    Across <strong>2,000 replayed episodes</strong>. Inventory Loss is discount given away
    plus scrap, so each column is shown split — the split is where the argument lives.
  </p>
</section>

<figure style="margin-top:18px">
  <div class="scroller"><table>
    <caption>Replay · 2,000 episodes</caption>
    <thead>
      <tr>
        <th>Measure</th>
        <th>Observed<br><span style="font-weight:400">what actually happened</span></th>
        <th>Legacy<br><span style="font-weight:400">under model</span></th>
        <th>Agent<br><span style="font-weight:400">under model</span></th>
      </tr>
    </thead>
    <tbody>
      <tr><td><strong>Inventory Loss</strong></td><td>₩17.11M</td><td>₩19.51M</td><td class="win">₩12.09M</td></tr>
      <tr><td>— discount given away</td><td>₩13.96M</td><td>₩13.27M</td><td class="good">₩5.26M</td></tr>
      <tr><td>— scrap</td><td>₩3.15M</td><td>₩6.24M</td><td class="loss">₩6.84M</td></tr>
      <tr><td>IL%</td><td>38.68%</td><td>45.14%</td><td class="win">28.77%</td></tr>
      <tr><td>Clearance</td><td>93.28%</td><td>77.58%</td><td class="loss">76.61%</td></tr>
      <tr><td>Mean discount</td><td>0.3094</td><td>0.2935</td><td class="good">0.1285</td></tr>
    </tbody>
  </table></div>
  <figcaption>
    Only the last two columns may be compared with each other. The first is the world, and it
    is in the table so that the second column can be judged against it — which is a question
    about the demand model, not about pricing.
  </figcaption>
</figure>

<section class="prose">
  <h3 style="margin-top:34px">The verdict, from the two comparable columns</h3>
</section>

<div class="math">legacy under model   ₩19.51M
agent  under model   ₩12.09M
                   ──────────
difference           −₩7.42M      =  −38.02% of the legacy arm
                                     clearance −0.97pp</div>

<section class="prose">
  <p style="margin-top:16px">
    <strong>₩7.42M less loss on 2,000 episodes.</strong> And the split says where every won
    of it came from: scrap went <em>up</em> ₩0.59M, so the entire gain — and more — came out
    of the discount line, which fell from ₩13.27M to ₩5.26M. The agent gave away
    <strong>₩8.01M less in markdown</strong> and paid ₩0.59M more in scrap to do it.
  </p>

  <p>
    Because that trade is real, the pilot reports <strong>clearance and scrap beside IL from
    day one</strong>. A system that improves its headline metric while quietly scrapping more
    food is a system whose reporting should show it, not hide it.
  </p>

  <h3 style="margin-top:34px">Why the tempting comparison would have misled us</h3>
  <p>
    Put the agent's ₩12.09M against the observed ₩17.11M and you get a
    <strong>−29.3%</strong> improvement. It is the comparison everyone reaches for, and it is
    not −38%. The two numbers differ because the demand model is not reality: it thinks the
    legacy rule would have lost ₩19.51M where the world actually lost ₩17.11M, and it clears
    77.58% where the world cleared 93.28%. Our model is pessimistic — by about
    <strong>14%</strong> on legacy's loss.
  </p>
  <p>
    That pessimism has nothing to do with the agent, and in the like-for-like comparison it
    cancels: both arms are penalised by it equally. Charge it to the agent instead and you
    are measuring a mixture of policy and model error, in a direction and size nobody can
    predict in advance — here it happens to <em>understate</em> the agent by nine points, but
    it could as easily have flattered it. That is why the rule is structural rather than a
    judgement call.
  </p>
</section>

<div class="callout">
  <h3>What replay cannot tell us — and this is the important limit</h3>
  <p>
    The agent chooses its prices by minimising expected loss under a particular set of
    assumptions: μ_ref for the demand level, r for the spread, ε for price response. The
    replay then simulates the outcome using <strong>those same assumptions</strong>. The agent
    is therefore playing against exactly its own model of the world — and in that world it is
    optimal <em>by construction</em>.
  </p>
  <p style="margin-top:14px">
    So a −38% result confirms something worth confirming: the solver is correct, the action
    set is safe, the plumbing works end to end, and the legacy rule really is a poor policy
    <em>for the world we believe in</em>. What it cannot do is test whether we believe in the
    right world. If real price response differs from the ε we assumed, both arms are wrong
    together, the comparison still cancels neatly, and the number on this page would not move
    at all.
  </p>
  <p style="margin-top:14px">
    <strong>Replay is a sanity check on the machine, never evidence that the policy wins in
    the market.</strong> Only the live A/B can say that, and it is designed on the assumption
    that this number might not survive contact.
  </p>
</div>
"""

PANELS["replay"] = P_REPLAY
