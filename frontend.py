import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from app import predict, df, bundle, lr, rf, feature_names

# --- Page config ---
st.set_page_config(
    page_title="LeadIQ — Conversion Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }

.stApp { background-color: #ffffff; }

.stTabs [data-baseweb="tab-list"] {
    background: #f0f2f6;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    color: #666680;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #00d4aa !important;
}

[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 16px;
    padding: 20px;
}
[data-testid="metric-container"] label {
    color: #666680 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00a882 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0099ff);
    color: #ffffff;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    font-size: 16px;
    padding: 16px 32px;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 212, 170, 0.3);
}

.stProgress > div > div {
    background: linear-gradient(90deg, #00d4aa, #0099ff) !important;
    border-radius: 4px !important;
}

.section-card {
    background: #f8f9fa;
    border: 1px solid #e8e8e8;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

.stat-pill {
    display: inline-block;
    background: #e8faf6;
    border: 1px solid #00d4aa;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #00a882;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
}

.insight-box {
    background: #f8f9fa;
    border-left: 3px solid #00d4aa;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# --- Matplotlib dark theme ---
mpl.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#444444',
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'text.color': '#262730',
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
})

# ── HEADER ──────────────────────────────────────────────────
st.markdown("""
<div style="padding: 40px 0 24px;">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <span style="font-size:32px;">🎯</span>
        <h1 style="margin:0; font-size:36px; font-weight:800;
            background: linear-gradient(135deg, #00d4aa, #0099ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            LeadIQ
        </h1>
    </div>
    <p style="color:#666680; font-size:15px; margin:0;">
        Conversion Intelligence Platform &nbsp;·&nbsp; Powered by ML
    </p>
    <div style="margin-top:16px;">
        <span class="stat-pill">9,240 Leads Trained</span>
        <span class="stat-pill">ROC AUC 0.899</span>
        <span class="stat-pill">83% Accuracy</span>
        <span class="stat-pill">Random Forest + LR Ensemble</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮  Predict Conversion", "📊  Data Analysis"])

# ============================================================
# TAB 1 — PREDICT
# ============================================================
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Activity Metrics")
        st.markdown("<br>", unsafe_allow_html=True)
        total_visits = st.slider("Total Visits", 0, 30, 5)
        st.markdown("<br>", unsafe_allow_html=True)
        time_spent = st.slider("Time on Website (sec)", 0, 2500, 800)
        st.markdown("<br>", unsafe_allow_html=True)
        page_views = st.slider("Page Views Per Visit", 0.0, 10.0, 3.0, step=0.5)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 👤 Lead Profile")
        st.markdown("<br>", unsafe_allow_html=True)
        lead_origin = st.selectbox("Lead Origin", [
            'Landing Page Submission', 'API', 'Lead Add Form',
            'Lead Import', 'Quick Add Form'
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        lead_source = st.selectbox("Lead Source", [
            'Google', 'Direct Traffic', 'Olark Chat', 'Organic Search',
            'Reference', 'Welingak Website', 'Referral Sites', 'Facebook', 'Unknown'
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        occupation = st.selectbox("Occupation", [
            'Unemployed', 'Working Professional', 'Student',
            'Businessman', 'Other', 'Unknown'
        ])
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 🔔 Engagement Signal")
        st.markdown("<br>", unsafe_allow_html=True)
        last_activity = st.selectbox("Last Activity", [
            'Email Opened', 'SMS Sent', 'Olark Chat Conversation',
            'Page Visited on Website', 'Converted to Lead',
            'Email Bounced', 'Email Link Clicked',
            'Form Submitted on Website', 'Unknown'
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        country = st.selectbox("Country", [
            'India', 'Unknown', 'United States', 'United Arab Emirates',
            'Singapore', 'Saudi Arabia', 'United Kingdom', 'Australia'
        ])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡ Run Prediction Engine", use_container_width=True):
        lr_prob, rf_prob, ensemble = predict({
            'TotalVisits': total_visits,
            'time_spent': time_spent,
            'page_views': page_views,
            'lead_origin': lead_origin,
            'lead_source': lead_source,
            'last_activity': last_activity,
            'occupation': occupation,
            'country': country,
        })
        pct = ensemble * 100

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Prediction Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ensemble Probability", f"{pct:.1f}%")
        m2.metric("Logistic Regression", f"{lr_prob*100:.1f}%")
        m3.metric("Random Forest", f"{rf_prob*100:.1f}%")
        m4.metric("Priority Tier",
                  "🔥 High" if pct >= 55 else "⚡ Medium" if pct >= 28 else "❄️ Low")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Conversion Likelihood — {pct:.1f}%**")
        st.progress(int(pct))

        st.markdown("<br>", unsafe_allow_html=True)
        if pct >= 55:
            st.success(f"✅ Strong conversion signal ({pct:.1f}%). Assign to senior sales rep. Follow up within 24 hours.")
        elif pct >= 28:
            st.warning(f"⚡ Moderate signal ({pct:.1f}%). Add to nurture campaign. Follow up within 3 days.")
        else:
            st.error(f"❌ Weak signal ({pct:.1f}%). Move to long-term drip campaign.")

# ============================================================
# TAB 2 — DATA ANALYSIS
# ============================================================
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Exploratory Data Analysis")
    st.markdown("<p style='color:#666680'>9,240 leads · 37 features · 38% base conversion rate</p>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2, gap="large")

    with r1c1:
        st.markdown("**Conversion Distribution**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        counts = df['Converted'].value_counts()
        bars = ax.bar(['Not Converted', 'Converted'], counts.values,
                      color=['#E24B4A', '#00d4aa'], width=0.5, edgecolor='none')
        ax.set_ylabel('Count', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{val:,}', ha='center', color='#e8e8f0', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with r1c2:
        st.markdown("**Conversion Rate by Lead Source**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        src = df.groupby('Lead Source')['Converted'].mean().sort_values(ascending=True).tail(8)
        ax.barh(src.index, src.values, color='#0099ff', edgecolor='none', height=0.6)
        ax.set_xlabel('Conversion Rate', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    r2c1, r2c2 = st.columns(2, gap="large")

    with r2c1:
        st.markdown("**Time on Website vs Conversion**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for label, color, name in zip([0, 1], ['#E24B4A', '#00d4aa'], ['Not Converted', 'Converted']):
            df[df['Converted'] == label]['Total Time Spent on Website'].plot(
                kind='hist', bins=30, alpha=0.7, ax=ax, label=name, color=color)
        ax.set_xlabel('Seconds', fontsize=11)
        ax.legend(fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with r2c2:
        st.markdown("**Conversion Rate by Occupation**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        occ = df.groupby('What is your current occupation')['Converted'].mean().sort_values(ascending=True)
        ax.barh(occ.index, occ.values, color='#00d4aa', edgecolor='none', height=0.6)
        ax.set_xlabel('Conversion Rate', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    r3c1, r3c2 = st.columns(2, gap="large")

    with r3c1:
        st.markdown("**Feature Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(6, 4))
        num_df = df[['TotalVisits', 'Total Time Spent on Website',
                     'Page Views Per Visit', 'Converted']].dropna()
        sns.heatmap(num_df.corr(), annot=True, fmt='.2f',
                    cmap=sns.diverging_palette(10, 165, as_cmap=True),
                    ax=ax, linewidths=0.5, linecolor='#1e1e2e',
                    annot_kws={'size': 12, 'color': '#e8e8f0'})
        ax.tick_params(colors='#888899', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with r3c2:
        st.markdown("**Top 10 Feature Importances (Random Forest)**")
        fi = bundle['feature_importances'][:10]
        names, scores = zip(*fi)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#00d4aa' if s == max(scores) else '#0099ff' if s > 0.05 else '#1e4a6e' for s in scores]
        ax.barh(list(names[::-1]), list(scores[::-1]), color=colors[::-1], edgecolor='none', height=0.6)
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Key insights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔑 Key Insights")
    st.markdown("<br>", unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3, gap="large")
    with i1:
        st.markdown("""
        <div class="insight-box">
            <p style="color:#00d4aa; font-weight:600; margin:0 0 6px; font-size:13px;">⏱️ TIME ON WEBSITE</p>
            <p style="color:#000000; margin:0; font-size:14px; line-height:1.6;">
                Strongest predictor at 44.9% importance. Leads spending 1000+ seconds convert at 2× the average rate.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div class="insight-box">
            <p style="color:#00d4aa; font-weight:600; margin:0 0 6px; font-size:13px;">💼 WORKING PROFESSIONALS</p>
            <p style="color:#000000; margin:0; font-size:14px; line-height:1.6;">
                Highest converting occupation segment. Prioritize outreach to employed leads over students.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with i3:
        st.markdown("""
        <div class="insight-box">
            <p style="color:#00d4aa; font-weight:600; margin:0 0 6px; font-size:13px;">📱 SMS ACTIVITY</p>
            <p style="color:#000000; margin:0; font-size:14px; line-height:1.6;">
                SMS Sent is the top activity signal at 6.6% importance. Follow up within 1 hour of SMS engagement.
            </p>
        </div>
        """, unsafe_allow_html=True)