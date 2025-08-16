import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import google.generativeai as genai
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import plotly.io as pio
import tempfile
import os
import requests
import warnings
warnings.filterwarnings("ignore", message=".*secrets.*")

DESIGN_SYSTEM = {
    'colors': {
        'primary': '#1E40AF',
        'secondary': '#059669',
        'accent': '#DC2626',
        'warning': '#D97706',
        'success': '#10B981',
        'background': '#F8FAFC',
        'text': '#1F2937',
        'border': '#E5E7EB'
    },
    'fonts': {
        'title': 'font-family: "Inter", sans-serif; font-weight: 700;',
        'subtitle': 'font-family: "Inter", sans-serif; font-weight: 600;',
        'body': 'font-family: "Inter", sans-serif; font-weight: 400;'
    }
}

st.set_page_config(
    page_title="Production Monitor with AI Insights | Nilsen Service & Consulting",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main-header {{
        background: linear-gradient(135deg, {DESIGN_SYSTEM['colors']['primary']} 0%, {DESIGN_SYSTEM['colors']['secondary']} 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }}
    .main-title {{
        {DESIGN_SYSTEM['fonts']['title']}
        font-size: 2.2rem;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .main-subtitle {{
        {DESIGN_SYSTEM['fonts']['body']}
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }}
    .metric-card {{
        background: white;
        border: 1px solid {DESIGN_SYSTEM['colors']['border']};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }}
    .section-header {{
        {DESIGN_SYSTEM['fonts']['subtitle']}
        color: {DESIGN_SYSTEM['colors']['text']};
        font-size: 1.4rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {DESIGN_SYSTEM['colors']['primary']};
    }}
    .chart-container {{
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
    .alert-success {{
        background: linear-gradient(135deg, {DESIGN_SYSTEM['colors']['success']}15, {DESIGN_SYSTEM['colors']['success']}25);
        border: 1px solid {DESIGN_SYSTEM['colors']['success']};
        border-radius: 8px;
        padding: 1rem;
        color: {DESIGN_SYSTEM['colors']['success']};
    }}
    .alert-warning {{
        background: linear-gradient(135deg, {DESIGN_SYSTEM['colors']['warning']}15, {DESIGN_SYSTEM['colors']['warning']}25);
        border: 1px solid {DESIGN_SYSTEM['colors']['warning']};
        border-radius: 8px;
        padding: 1rem;
        color: {DESIGN_SYSTEM['colors']['warning']};
    }}
    .stButton > button {{
        background: {DESIGN_SYSTEM['colors']['primary']};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    .stDownloadButton > button {{
        background: {DESIGN_SYSTEM['colors']['primary']} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def init_ai():
    """Initialize AI model with proper error handling for secrets"""
    try:
        # Try to get API key from Streamlit secrets
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except (FileNotFoundError, KeyError, AttributeError):
        # If secrets file doesn't exist or key not found, try environment variable
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            st.error(f"AI configuration failed: {str(e)}")
            return None
    return None

@st.cache_data
def load_preset_data(year):
    urls = {
        "2024": "https://huggingface.co/spaces/entropy25/production-data-analysis/resolve/main/2024.csv",
        "2025": "https://huggingface.co/spaces/entropy25/production-data-analysis/resolve/main/2025.csv"
    }
    try:
        if year in urls:
            response = requests.get(urls[year], timeout=10)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text), sep='\t')
            df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
            df['day_name'] = df['date'].dt.day_name()
            return df
        else:
            return generate_sample_data(year)
    except Exception as e:
        st.warning(f"Could not load remote {year} data: {str(e)}. Loading sample data instead.")
        return generate_sample_data(year)

def generate_sample_data(year):
    np.random.seed(42 if year == "2024" else 84)
    start_date = f"01/01/{year}"
    end_date = f"12/31/{year}"
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    weekdays = dates[dates.weekday < 5]
    data = []
    materials = ['steel', 'aluminum', 'plastic', 'copper']
    shifts = ['day', 'night']
    for date in weekdays:
        for material in materials:
            for shift in shifts:
                base_weight = {
                    'steel': 1500,
                    'aluminum': 800,
                    'plastic': 600,
                    'copper': 400
                }[material]
                weight = base_weight + np.random.normal(0, base_weight * 0.2)
                weight = max(weight, base_weight * 0.3)
                data.append({
                    'date': date.strftime('%m/%d/%Y'),
                    'weight_kg': round(weight, 1),
                    'material_type': material,
                    'shift': shift
                })
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df['day_name'] = df['date'].dt.day_name()
    return df

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep='\t')
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df['day_name'] = df['date'].dt.day_name()
    return df

def get_material_stats(df):
    stats = {}
    total = df['weight_kg'].sum()
    total_work_days = df['date'].nunique()
    for material in df['material_type'].unique():
        data = df[df['material_type'] == material]
        work_days = data['date'].nunique()
        daily_avg = data.groupby('date')['weight_kg'].sum().mean()
        stats[material] = {
            'total': data['weight_kg'].sum(),
            'percentage': (data['weight_kg'].sum() / total) * 100,
            'daily_avg': daily_avg,
            'work_days': work_days,
            'records': len(data)
        }
    stats['_total_'] = {
        'total': total,
        'percentage': 100.0,
        'daily_avg': df.groupby('date')['weight_kg'].sum().mean(),
        'work_days': total_work_days,
        'records': len(df)
    }
    return stats

def get_chart_theme():
    return {
        'layout': {
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'font': {'family': 'Inter, sans-serif', 'color': DESIGN_SYSTEM['colors']['text']},
            'colorway': [DESIGN_SYSTEM['colors']['primary'], DESIGN_SYSTEM['colors']['secondary'], 
                        DESIGN_SYSTEM['colors']['accent'], DESIGN_SYSTEM['colors']['warning']],
            'margin': {'t': 60, 'b': 40, 'l': 40, 'r': 40}
        }
    }

def create_total_production_chart(df, time_period='daily'):
    if time_period == 'daily':
        grouped = df.groupby('date')['weight_kg'].sum().reset_index()
        fig = px.line(grouped, x='date', y='weight_kg', 
                     title='Total Production Trend',
                     labels={'weight_kg': 'Weight (kg)', 'date': 'Date'})
    elif time_period == 'weekly':
        df_copy = df.copy()
        df_copy['week'] = df_copy['date'].dt.isocalendar().week
        df_copy['year'] = df_copy['date'].dt.year
        grouped = df_copy.groupby(['year', 'week'])['weight_kg'].sum().reset_index()
        grouped['week_label'] = grouped['year'].astype(str) + '-W' + grouped['week'].astype(str)
        fig = px.bar(grouped, x='week_label', y='weight_kg',
                    title='Total Production Trend (Weekly)',
                    labels={'weight_kg': 'Weight (kg)', 'week_label': 'Week'})
    else:
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.to_period('M')
        grouped = df_copy.groupby('month')['weight_kg'].sum().reset_index()
        grouped['month'] = grouped['month'].astype(str)
        fig = px.bar(grouped, x='month', y='weight_kg',
                    title='Total Production Trend (Monthly)',
                    labels={'weight_kg': 'Weight (kg)', 'month': 'Month'})
    fig.update_layout(**get_chart_theme()['layout'], height=400, showlegend=False)
    return fig

def create_materials_trend_chart(df, time_period='daily', selected_materials=None):
    df_copy = df.copy()
    if selected_materials:
        df_copy = df_copy[df_copy['material_type'].isin(selected_materials)]
    if time_period == 'daily':
        grouped = df_copy.groupby(['date', 'material_type'])['weight_kg'].sum().reset_index()
        fig = px.line(grouped, x='date', y='weight_kg', color='material_type',
                     title='Materials Production Trends',
                     labels={'weight_kg': 'Weight (kg)', 'date': 'Date', 'material_type': 'Material'})
    elif time_period == 'weekly':
        df_copy['week'] = df_copy['date'].dt.isocalendar().week
        df_copy['year'] = df_copy['date'].dt.year
        grouped = df_copy.groupby(['year', 'week', 'material_type'])['weight_kg'].sum().reset_index()
        grouped['week_label'] = grouped['year'].astype(str) + '-W' + grouped['week'].astype(str)
        fig = px.bar(grouped, x='week_label', y='weight_kg', color='material_type',
                    title='Materials Production Trends (Weekly)',
                    labels={'weight_kg': 'Weight (kg)', 'week_label': 'Week', 'material_type': 'Material'})
    else:
        df_copy['month'] = df_copy['date'].dt.to_period('M')
        grouped = df_copy.groupby(['month', 'material_type'])['weight_kg'].sum().reset_index()
        grouped['month'] = grouped['month'].astype(str)
        fig = px.bar(grouped, x='month', y='weight_kg', color='material_type',
                    title='Materials Production Trends (Monthly)',
                    labels={'weight_kg': 'Weight (kg)', 'month': 'Month', 'material_type': 'Material'})
    fig.update_layout(**get_chart_theme()['layout'], height=400)
    return fig

def create_shift_trend_chart(df, time_period='daily'):
    if time_period == 'daily':
        grouped = df.groupby(['date', 'shift'])['weight_kg'].sum().reset_index()
        pivot_data = grouped.pivot(index='date', columns='shift', values='weight_kg').fillna(0)
        fig = go.Figure()
        if 'day' in pivot_data.columns:
            fig.add_trace(go.Bar(
                x=pivot_data.index, y=pivot_data['day'], name='Day Shift',
                marker_color=DESIGN_SYSTEM['colors']['warning'],
                text=pivot_data['day'].round(0), textposition='inside'
            ))
        if 'night' in pivot_data.columns:
            fig.add_trace(go.Bar(
                x=pivot_data.index, y=pivot_data['night'], name='Night Shift',
                marker_color=DESIGN_SYSTEM['colors']['primary'],
                base=pivot_data['day'] if 'day' in pivot_data.columns else 0,
                text=pivot_data['night'].round(0), textposition='inside'
            ))
        fig.update_layout(
            **get_chart_theme()['layout'],
            title='Daily Shift Production Trends (Stacked)',
            xaxis_title='Date', yaxis_title='Weight (kg)',
            barmode='stack', height=400, showlegend=True
        )
    else:
        grouped = df.groupby(['date', 'shift'])['weight_kg'].sum().reset_index()
        fig = px.bar(grouped, x='date', y='weight_kg', color='shift',
                    title=f'{time_period.title()} Shift Production Trends',
                    barmode='stack')
        fig.update_layout(**get_chart_theme()['layout'], height=400)
    return fig

def detect_outliers(df):
    outliers = {}
    for material in df['material_type'].unique():
        material_data = df[df['material_type'] == material]
        data = material_data['weight_kg']
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outlier_mask = (data < lower) | (data > upper)
        outlier_dates = material_data[outlier_mask]['date'].dt.strftime('%Y-%m-%d').tolist()
        outliers[material] = {
            'count': len(outlier_dates),
            'range': f"{lower:.0f} - {upper:.0f} kg",
            'dates': outlier_dates
        }
    return outliers

def generate_ai_summary(model, df, stats, outliers):
    if not model:
        return "AI analysis unavailable - Google API key not configured. Please set the GOOGLE_API_KEY environment variable or in Streamlit secrets to enable AI insights."
    try:
        materials = [k for k in stats.keys() if k != '_total_']
        context_parts = [
            "# Production Data Analysis Context",
            f"## Overview",
            f"- Total Production: {stats['_total_']['total']:,.0f} kg",
            f"- Production Period: {stats['_total_']['work_days']} working days", 
            f"- Daily Average: {stats['_total_']['daily_avg']:,.0f} kg",
            f"- Materials Tracked: {len(materials)}",
            "",
            "## Material Breakdown:"
        ]
        for material in materials:
            info = stats[material]
            context_parts.append(f"- {material.title()}: {info['total']:,.0f} kg ({info['percentage']:.1f}%), avg {info['daily_avg']:,.0f} kg/day")
        daily_data = df.groupby('date')['weight_kg'].sum()
        trend_direction = "increasing" if daily_data.iloc[-1] > daily_data.iloc[0] else "decreasing"
        volatility = daily_data.std() / daily_data.mean() * 100
        context_parts.extend([
            "",
            "## Trend Analysis:",
            f"- Overall trend: {trend_direction}",
            f"- Production volatility: {volatility:.1f}% coefficient of variation",
            f"- Peak production: {daily_data.max():,.0f} kg",
            f"- Lowest production: {daily_data.min():,.0f} kg"
        ])
        total_outliers = sum(info['count'] for info in outliers.values())
        context_parts.extend([
            "",
            "## Quality Control:",
            f"- Total outliers detected: {total_outliers}",
            f"- Materials with quality issues: {sum(1 for info in outliers.values() if info['count'] > 0)}"
        ])
        if 'shift' in df.columns:
            shift_stats = df.groupby('shift')['weight_kg'].sum()
            context_parts.extend([
                "",
                "## Shift Performance:",
                f"- Day shift: {shift_stats.get('day', 0):,.0f} kg",
                f"- Night shift: {shift_stats.get('night', 0):,.0f} kg"
            ])
        context_text = "\n".join(context_parts)
        prompt = f"""
{context_text}

As an expert AI analyst embedded within the "Production Monitor with AI Insights" platform, provide a comprehensive analysis based on the data provided. Your tone should be professional and data-driven. Your primary goal is to highlight how the platform's features reveal critical insights.

Structure your response in the following format:

**PRODUCTION ASSESSMENT**
Evaluate the overall production status (Excellent/Good/Needs Attention). Briefly justify your assessment using key metrics from the data summary.

**KEY FINDINGS**
Identify 3-4 of the most important insights. For each finding, explicitly mention the platform feature that made the discovery possible. Use formats like "(revealed by the 'Quality Check' module)" or "(visualized in the 'Production Trend' chart)".

Example Finding format:
• Finding X: [Your insight, e.g., "Liquid-Ctu production shows high volatility..."] (as identified by the 'Materials Analysis' view).

**RECOMMENDATIONS**
Provide 2-3 actionable recommendations. Frame these as steps the management can take, encouraging them to use the platform for further investigation.

Example Recommendation format:
• Recommendation Y: [Your recommendation, e.g., "Investigate the root causes of the 11 outliers..."] We recommend using the platform's interactive charts to drill down into the specific dates identified by the 'Quality Check' module.

Keep the entire analysis concise and under 300 words.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI analysis error: {str(e)}"

def query_ai(model, stats, question, df=None):
    if not model:
        return "AI assistant not available - Please configure Google API key"
    context_parts = [
        "Production Data Summary:",
        *[f"- {mat.title()}: {info['total']:,.0f}kg ({info['percentage']:.1f}%)" 
          for mat, info in stats.items() if mat != '_total_'],
        f"\nTotal Production: {stats['_total_']['total']:,.0f}kg across {stats['_total_']['work_days']} work days"
    ]
    if df is not None:
        available_cols = list(df.columns)
        context_parts.append(f"\nAvailable data fields: {', '.join(available_cols)}")
        if 'shift' in df.columns:
            shift_stats = df.groupby('shift')['weight_kg'].sum()
            context_parts.append(f"Shift breakdown: {dict(shift_stats)}")
        if 'day_name' in df.columns:
            day_stats = df.groupby('day_name')['weight_kg'].mean()
            context_parts.append(f"Average daily production: {dict(day_stats.round(0))}")
    context = "\n".join(context_parts) + f"\n\nQuestion: {question}\nAnswer based on available data:"
    try:
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

def save_plotly_as_image(fig, filename):
    try:
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        theme = get_chart_theme()['layout'].copy()
        theme.update({
            'font': dict(size=12, family="Arial"),
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': dict(t=50, b=40, l=40, r=40)
        })
        fig.update_layout(**theme)
        try:
            pio.write_image(fig, filepath, format='png', width=800, height=400, scale=2, engine='kaleido')
            if os.path.exists(filepath):
                return filepath
        except:
            pass
        return None
    except Exception as e:
        return None

def create_pdf_charts(df, stats):
    charts = {}
    try:
        materials = [k for k in stats.keys() if k != '_total_']
        values = [stats[mat]['total'] for mat in materials]
        labels = [mat.replace('_', ' ').title() for mat in materials]
        if len(materials) > 0 and len(values) > 0:
            try:
                fig_pie = px.pie(values=values, names=labels, title="Production Distribution by Material")
                charts['pie'] = save_plotly_as_image(fig_pie, "distribution.png")
            except:
                pass
        if len(df) > 0:
            try:
                daily_data = df.groupby('date')['weight_kg'].sum().reset_index()
                if len(daily_data) > 0:
                    fig_trend = px.line(daily_data, x='date', y='weight_kg', title="Daily Production Trend",
                                        labels={'date': 'Date', 'weight_kg': 'Weight (kg)'},
                                        color_discrete_sequence=[DESIGN_SYSTEM['colors']['primary']])
                    charts['trend'] = save_plotly_as_image(fig_trend, "trend.png")
            except:
                pass
        if len(materials) > 0 and len(values) > 0:
            try:
                fig_bar = px.bar(x=labels, y=values, title="Production by Material Type",
                                 labels={'x': 'Material Type', 'y': 'Weight (kg)'},
                                 color_discrete_sequence=[DESIGN_SYSTEM['colors']['primary']])
                charts['bar'] = save_plotly_as_image(fig_bar, "materials.png")
            except:
                pass
        if 'shift' in df.columns and len(df) > 0:
            try:
                shift_data = df.groupby('shift')['weight_kg'].sum().reset_index()
                if len(shift_data) > 0 and shift_data['weight_kg'].sum() > 0:
                    fig_shift = px.pie(shift_data, values='weight_kg', names='shift', title="Production by Shift")
                    charts['shift'] = save_plotly_as_image(fig_shift, "shifts.png")
            except:
                pass
    except Exception as e:
        pass
    return charts

def create_enhanced_pdf_report(df, stats, outliers, model=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        textColor=colors.darkblue
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=20,
        textColor=colors.darkblue
    )
    ai_style = ParagraphStyle(
        'AIStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        leftIndent=20,
        textColor=colors.darkgreen
    )
    elements.append(Spacer(1, 100))
    elements.append(Paragraph("Production Monitor with AI Insights", title_style))
    elements.append(Paragraph("Comprehensive Production Analysis Report", styles['Heading3']))
    elements.append(Spacer(1, 50))
    report_info = f"""
    <para alignment="center">
    <b>Nilsen Service &amp; Consulting AS</b><br/>
    Production Analytics Division<br/><br/>
    <b>Report Period:</b> {df['date'].min().strftime('%B %d, %Y')} - {df['date'].max().strftime('%B %d, %Y')}<br/>
    <b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}<br/>
    <b>Total Records:</b> {len(df):,}
    </para>
    """
    elements.append(Paragraph(report_info, styles['Normal']))
    elements.append(PageBreak())
    elements.append(Paragraph("Executive Summary", subtitle_style))
    total_production = stats['_total_']['total']
    work_days = stats['_total_']['work_days']
    daily_avg = stats['_total_']['daily_avg']
    exec_summary = f"""
    <para>
    This report analyzes production data spanning <b>{work_days} working days</b>. 
    Total output achieved: <b>{total_production:,.0f} kg</b> with an average 
    daily production of <b>{daily_avg:,.0f} kg</b>.
    <br/><br/>
    <b>Key Highlights:</b><br/>
    • Total production: {total_production:,.0f} kg<br/>
    • Daily average: {daily_avg:,.0f} kg<br/>
    • Materials tracked: {len([k for k in stats.keys() if k != '_total_'])}<br/>
    • Data quality: {len(df):,} records processed
    </para>
    """
    elements.append(Paragraph(exec_summary, styles['Normal']))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Production Summary", styles['Heading3']))
    summary_data = [['Material Type', 'Total (kg)', 'Share (%)', 'Daily Avg (kg)']]
    for material, info in stats.items():
        if material != '_total_':
            summary_data.append([
                material.replace('_', ' ').title(),
                f"{info['total']:,.0f}",
                f"{info['percentage']:.1f}%",
                f"{info['daily_avg']:,.0f}"
            ])
    summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    elements.append(summary_table)
    elements.append(PageBreak())
    elements.append(Paragraph("Production Analysis Charts", subtitle_style))
    try:
        charts = create_pdf_charts(df, stats)
    except:
        charts = {}
    charts_added = False
    chart_insights = {
        'pie': "Material distribution shows production allocation across different materials. Balanced distribution indicates diversified production capabilities.",
        'trend': "Production trend reveals operational patterns and seasonal variations. Consistent trends suggest stable operational efficiency.",
        'bar': "Material comparison highlights performance differences and production capacities. Top performers indicate optimization opportunities.",
        'shift': "Shift analysis reveals operational efficiency differences between day and night operations. Balance indicates effective resource utilization."
    }
    for chart_type, chart_title in [
        ('pie', "Production Distribution"),
        ('trend', "Production Trend"), 
        ('bar', "Material Comparison"),
        ('shift', "Shift Analysis")
    ]:
        chart_path = charts.get(chart_type)
        if chart_path and os.path.exists(chart_path):
            try:
                elements.append(Paragraph(chart_title, styles['Heading3']))
                elements.append(Image(chart_path, width=6*inch, height=3*inch))
                insight_text = f"<i>Analysis: {chart_insights.get(chart_type, 'Chart analysis not available.')}</i>"
                elements.append(Paragraph(insight_text, ai_style))
                elements.append(Spacer(1, 20))
                charts_added = True
            except Exception as e:
                pass
    if not charts_added:
        elements.append(Paragraph("Charts Generation Failed", styles['Heading3']))
        elements.append(Paragraph("Production Data Summary:", styles['Normal']))
        for material, info in stats.items():
            if material != '_total_':
                summary_text = f"• {material.replace('_', ' ').title()}: {info['total']:,.0f} kg ({info['percentage']:.1f}%)"
                elements.append(Paragraph(summary_text, styles['Normal']))
        elements.append(Spacer(1, 20))
    elements.append(PageBreak())
    elements.append(Paragraph("Quality Control Analysis", subtitle_style))
    quality_data = [['Material', 'Outliers', 'Normal Range (kg)', 'Status']]
    for material, info in outliers.items():
        if info['count'] == 0:
            status = "GOOD"
        elif info['count'] <= 3:
            status = "MONITOR"
        else:
            status = "ATTENTION"
        quality_data.append([
            material.replace('_', ' ').title(),
            str(info['count']),
            info['range'],
            status
        ])
    quality_table = Table(quality_data, colWidths=[2*inch, 1*inch, 2*inch, 1.5*inch])
    quality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    elements.append(quality_table)
    if model:
        elements.append(PageBreak())
        elements.append(Paragraph("AI Intelligent Analysis", subtitle_style))
        try:
            ai_analysis = generate_ai_summary(model, df, stats, outliers)
        except:
            ai_analysis = "AI analysis temporarily unavailable."
        ai_paragraphs = ai_analysis.split('\n\n')
        for paragraph in ai_paragraphs:
            if paragraph.strip():
                formatted_text = paragraph.replace('**', '<b>', 1).replace('**', '</b>', 1) \
                                            .replace('•', '  •') \
                                            .replace('\n', '<br/>')
                elements.append(Paragraph(formatted_text, styles['Normal']))
                elements.append(Spacer(1, 8))
    else:
        elements.append(PageBreak())
        elements.append(Paragraph("AI Analysis", subtitle_style))
        elements.append(Paragraph("AI analysis unavailable - Google API key not configured. Please set the GOOGLE_API_KEY environment variable or configure it in Streamlit secrets to enable intelligent insights.", styles['Normal']))
    elements.append(Spacer(1, 30))
    footer_text = f"""
    <para alignment="center">
    <i>This report was generated by Production Monitor System<br/>
    Nilsen Service &amp; Consulting AS - Production Analytics Division<br/>
    Report contains {len(df):,} data records across {stats['_total_']['work_days']} working days</i>
    </para>
    """
    elements.append(Paragraph(footer_text, styles['Normal']))
    doc.build(elements)
    buffer.seek(0)
    return buffer

def create_csv_export(df, stats):
    summary_df = pd.DataFrame([
        {
            'Material': material.replace('_', ' ').title(),
            'Total_kg': info['total'],
            'Percentage': info['percentage'],
            'Daily_Average_kg': info['daily_avg'],
            'Work_Days': info['work_days'],
            'Records_Count': info['records']
        }
        for material, info in stats.items() if material != '_total_'
    ])
    return summary_df

def add_export_section(df, stats, outliers, model):
    st.markdown('<div class="section-header">📄 Export Reports</div>', unsafe_allow_html=True)
    if 'export_ready' not in st.session_state:
        st.session_state.export_ready = False
    if 'pdf_buffer' not in st.session_state:
        st.session_state.pdf_buffer = None
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Generate PDF Report with AI", key="generate_pdf_btn", type="primary"):
            try:
                with st.spinner("Generating PDF with AI analysis..."):
                    st.session_state.pdf_buffer = create_enhanced_pdf_report(df, stats, outliers, model)
                    st.session_state.export_ready = True
                st.success("✅ PDF report with AI analysis generated successfully!")
            except Exception as e:
                st.error(f"❌ PDF generation failed: {str(e)}")
                st.session_state.export_ready = False
        if st.session_state.export_ready and st.session_state.pdf_buffer:
            st.download_button(
                label="💾 Download PDF Report",
                data=st.session_state.pdf_buffer,
                file_name=f"production_report_ai_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key="download_pdf_btn"
            )
    with col2:
        if st.button("Generate CSV Summary", key="generate_csv_btn", type="primary"):
            try:
                st.session_state.csv_data = create_csv_export(df, stats)
                st.success("✅ CSV summary generated successfully!")
            except Exception as e:
                st.error(f"❌ CSV generation failed: {str(e)}")
        if st.session_state.csv_data is not None:
            csv_string = st.session_state.csv_data.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV Summary",
                data=csv_string,
                file_name=f"production_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="download_csv_btn"
            )
    with col3:
        csv_string = df.to_csv(index=False)
        st.download_button(
            label="Download Raw Data",
            data=csv_string,
            file_name=f"raw_production_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="download_raw_btn"
        )

def main():
    load_css()
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🏭 Production Monitor with AI Insights</div>
        <div class="main-subtitle">Nilsen Service & Consulting AS | Real-time Production Analytics & Recommendations</div>
    </div>
    """, unsafe_allow_html=True)
    model = init_ai()
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'current_stats' not in st.session_state:
        st.session_state.current_stats = None
    with st.sidebar:
        st.markdown("### 📊 Data Source")
        uploaded_file = st.file_uploader("Upload Production Data", type=['csv'])
        st.markdown("---")
        st.markdown("### 📊 Quick Load")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 2024 Data", type="primary", key="load_2024"):
                st.session_state.load_preset = "2024"
        with col2:
            if st.button("📊 2025 Data", type="primary", key="load_2025"):
                st.session_state.load_preset = "2025"
        st.markdown("---")
        st.markdown("""
        **Expected TSV format:**
        - `date`: MM/DD/YYYY
        - `weight_kg`: Production weight
        - `material_type`: Material category
        - `shift`: day/night (optional)
        """)
        if model:
            st.success("🤖 AI Assistant Ready")
        else:
            st.warning("⚠️ AI Assistant Unavailable")
            st.info("To enable AI features, set GOOGLE_API_KEY as environment variable or in Streamlit secrets")
    df = st.session_state.current_df
    stats = st.session_state.current_stats
    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            stats = get_material_stats(df)
            st.session_state.current_df = df
            st.session_state.current_stats = stats
            st.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading uploaded file: {str(e)}")
    elif 'load_preset' in st.session_state:
        year = st.session_state.load_preset
        try:
            with st.spinner(f"Loading {year} data..."):
                df = load_preset_data(year)
            if df is not None:
                stats = get_material_stats(df)
                st.session_state.current_df = df
                st.session_state.current_stats = stats
                st.success(f"✅ {year} data loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading {year} data: {str(e)}")
        finally:
            del st.session_state.load_preset
    if df is not None and stats is not None:
        st.markdown('<div class="section-header">📋 Material Overview</div>', unsafe_allow_html=True)
        materials = [k for k in stats.keys() if k != '_total_']
        cols = st.columns(4)
        for i, material in enumerate(materials[:3]):
            info = stats[material]
            with cols[i]:
                st.metric(
                    label=material.replace('_', ' ').title(),
                    value=f"{info['total']:,.0f} kg",
                    delta=f"{info['percentage']:.1f}% of total"
                )
                st.caption(f"Daily avg: {info['daily_avg']:,.0f} kg")
        if len(materials) >= 3:
            total_info = stats['_total_']
            with cols[3]:
                st.metric(
                    label="Total Production",
                    value=f"{total_info['total']:,.0f} kg",
                    delta="100% of total"
                )
                st.caption(f"Daily avg: {total_info['daily_avg']:,.0f} kg")
        st.markdown('<div class="section-header">📊 Production Trends</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col2:
            time_view = st.selectbox("Time Period", ["daily", "weekly", "monthly"], key="time_view_select")
        with col1:
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                total_chart = create_total_production_chart(df, time_view)
                st.plotly_chart(total_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🏷️ Materials Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col2:
            selected_materials = st.multiselect(
                "Select Materials", 
                options=materials, 
                default=materials,
                key="materials_select"
            )
        with col1:
            if selected_materials:
                with st.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    materials_chart = create_materials_trend_chart(df, time_view, selected_materials)
                    st.plotly_chart(materials_chart, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        if 'shift' in df.columns:
            st.markdown('<div class="section-header">🌓 Shift Analysis</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                shift_chart = create_shift_trend_chart(df, time_view)
                st.plotly_chart(shift_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">⚠️ Quality Check</div>', unsafe_allow_html=True)
        outliers = detect_outliers(df)
        cols = st.columns(len(outliers))
        for i, (material, info) in enumerate(outliers.items()):
            with cols[i]:
                if info['count'] > 0:
                    if len(info['dates']) <= 5:
                        dates_str = ", ".join(info['dates'])
                    else:
                        dates_str = f"{', '.join(info['dates'][:3])}, +{len(info['dates'])-3} more"
                    st.markdown(f'<div class="alert-warning"><strong>{material.title()}</strong><br>{info["count"]} outliers detected<br>Normal range: {info["range"]}<br><small>Dates: {dates_str}</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-success"><strong>{material.title()}</strong><br>All values normal</div>', unsafe_allow_html=True)
        add_export_section(df, stats, outliers, model)
        if model:
            st.markdown('<div class="section-header">🤖 AI Insights</div>', unsafe_allow_html=True)
            quick_questions = [
                "How does production distribution on weekdays compare to weekends?",
                "Which material exhibits the most volatility in our dataset?",
                "To improve stability, which material or shift needs immediate attention?"
            ]
            cols = st.columns(len(quick_questions))
            for i, q in enumerate(quick_questions):
                with cols[i]:
                    if st.button(q, key=f"ai_q_{i}"):
                        with st.spinner("Analyzing..."):
                            answer = query_ai(model, stats, q, df)
                            st.info(answer)
            custom_question = st.text_input("Ask about your production data:", 
                                            placeholder="e.g., 'Compare steel vs aluminum last month'",
                                            key="custom_ai_question")
            if custom_question and st.button("Ask AI", key="ask_ai_btn"):
                with st.spinner("Analyzing..."):
                    answer = query_ai(model, stats, custom_question, df)
                    st.success(f"**Q:** {custom_question}")
                    st.write(f"**A:** {answer}")
        else:
            st.markdown('<div class="section-header">🤖 AI Configuration</div>', unsafe_allow_html=True)
            st.info("""
            **AI Assistant is currently unavailable.**
            
            To enable AI features, you need to configure your Google AI API key:
            
            **Option 1: Environment Variable**
            ```bash
            export GOOGLE_API_KEY="your_api_key_here"
            ```
            
            **Option 2: Streamlit Secrets**
            Create `.streamlit/secrets.toml`:
            ```toml
            GOOGLE_API_KEY = "your_api_key_here"
            ```
            
            **Option 3: Azure App Service**
            Set environment variable in Azure portal under Configuration > Application settings.
            """)
    else:
        st.markdown('<div class="section-header">📖 How to Use This Platform</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🚀 Quick Start
            1. Upload your TSV data in the sidebar
            2. Or click Quick Load buttons for preset data
            3. View production by material type
            4. Analyze trends (daily/weekly/monthly)
            5. Check anomalies in Quality Check
            6. Export reports (PDF with AI, CSV)
            7. Ask the AI assistant for insights
            """)
        with col2:
            st.markdown("""
            ### 📊 Key Features
            - Real-time interactive charts
            - One-click preset data loading
            - Time-period comparisons
            - Shift performance analysis
            - Outlier detection with dates
            - AI-powered PDF reports
            - Intelligent recommendations
            """)
        st.info("📁 Ready to start? Upload your production data or use Quick Load buttons to begin analysis!")

if __name__ == "__main__":
    main()
