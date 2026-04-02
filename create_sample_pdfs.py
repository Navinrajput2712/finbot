from fpdf import FPDF
import os

os.makedirs("data/knowledge_base", exist_ok=True)

# ── PDF 1 — Investing Guide ──────────────────────────────
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)
content1 = [
    "INVESTING GUIDE FOR BEGINNERS",
    "",
    "1. MUTUAL FUNDS",
    "A mutual fund pools money from many investors to purchase securities.",
    "Types: Equity funds, Debt funds, Hybrid funds, Index funds.",
    "SIP (Systematic Investment Plan) allows investing fixed amounts monthly.",
    "Minimum SIP amount is usually Rs 500 per month.",
    "ELSS (Equity Linked Savings Scheme) offers tax benefits under Section 80C.",
    "",
    "2. STOCKS",
    "Stocks represent ownership in a company listed on NSE or BSE.",
    "Nifty 50 tracks the top 50 companies on the National Stock Exchange.",
    "Sensex tracks the top 30 companies on the Bombay Stock Exchange.",
    "Long term capital gains tax is 10% above Rs 1 lakh profit.",
    "Short term capital gains tax is 15% for equity investments.",
    "",
    "3. PORTFOLIO ALLOCATION",
    "Rule of 100: Subtract your age from 100 for equity allocation.",
    "Example: Age 30 means 70% equity, 30% debt allocation.",
    "Emergency fund should be kept in liquid funds or savings account.",
    "Diversification reduces risk across different asset classes.",
    "",
    "4. SIP CALCULATION",
    "To build Rs 1 crore corpus in 20 years at 12% return:",
    "Monthly SIP required = Rs 10,000 approximately.",
    "Power of compounding works best with long investment horizon.",
    "Start investing early to maximize compounding benefits.",
]
for line in content1:
    pdf.cell(0, 10, line, ln=True)
pdf.output("data/knowledge_base/investing_guide.pdf")
print("Created: investing_guide.pdf")

# ── PDF 2 — Taxation Guide ───────────────────────────────
pdf2 = FPDF()
pdf2.add_page()
pdf2.set_font("Helvetica", size=12)
content2 = [
    "INCOME TAX GUIDE FOR INDIVIDUALS - INDIA",
    "",
    "1. TAX SLABS FY 2024-25 (NEW REGIME)",
    "Income up to Rs 3,00,000 - No tax",
    "Income Rs 3,00,001 to Rs 6,00,000 - 5% tax",
    "Income Rs 6,00,001 to Rs 9,00,000 - 10% tax",
    "Income Rs 9,00,001 to Rs 12,00,000 - 15% tax",
    "Income Rs 12,00,001 to Rs 15,00,000 - 20% tax",
    "Income above Rs 15,00,000 - 30% tax",
    "",
    "2. TAX SLABS FY 2024-25 (OLD REGIME)",
    "Income up to Rs 2,50,000 - No tax",
    "Income Rs 2,50,001 to Rs 5,00,000 - 5% tax",
    "Income Rs 5,00,001 to Rs 10,00,000 - 20% tax",
    "Income above Rs 10,00,000 - 30% tax",
    "",
    "3. SECTION 80C DEDUCTIONS (OLD REGIME)",
    "Maximum deduction limit under Section 80C is Rs 1,50,000.",
    "Eligible investments: PPF, ELSS, NSC, Tax Saver FD, LIC Premium.",
    "EPF contribution by employee qualifies under 80C.",
    "Home loan principal repayment qualifies under 80C.",
    "Tuition fees for children qualifies under 80C.",
    "",
    "4. OTHER DEDUCTIONS",
    "Section 80D: Health insurance premium up to Rs 25,000.",
    "Section 80D: Senior citizen health insurance up to Rs 50,000.",
    "Section 24b: Home loan interest deduction up to Rs 2,00,000.",
    "HRA exemption available for salaried employees paying rent.",
    "Standard deduction of Rs 50,000 for salaried individuals.",
    "",
    "5. ITR FILING",
    "ITR filing deadline for individuals is July 31 each year.",
    "ITR-1 (Sahaj) for salary income up to Rs 50 lakh.",
    "ITR-2 for capital gains and multiple income sources.",
    "Late filing penalty is Rs 5,000 after due date.",
]
for line in content2:
    pdf2.cell(0, 10, line, ln=True)
pdf2.output("data/knowledge_base/taxation_guide.pdf")
print("Created: taxation_guide.pdf")

# ── PDF 3 — Personal Finance Guide ──────────────────────
pdf3 = FPDF()
pdf3.add_page()
pdf3.set_font("Helvetica", size=12)
content3 = [
    "PERSONAL FINANCE AND LOANS GUIDE - INDIA",
    "",
    "1. BUDGETING",
    "50/30/20 Rule: 50% needs, 30% wants, 20% savings.",
    "Emergency fund should cover 6 months of expenses.",
    "Track expenses using apps like Money Manager or Walnut.",
    "Avoid lifestyle inflation when income increases.",
    "",
    "2. HOME LOAN",
    "EMI Formula: EMI = P x R x (1+R)^N / ((1+R)^N - 1)",
    "P = Principal loan amount",
    "R = Monthly interest rate (Annual rate divided by 12)",
    "N = Loan tenure in months",
    "Example: Rs 50 lakh loan at 8.5% for 20 years = EMI Rs 43,391",
    "Home loan interest deduction up to Rs 2 lakh under Section 24b.",
    "PMAY subsidy available for first time home buyers.",
    "",
    "3. CIBIL SCORE",
    "CIBIL score ranges from 300 to 900.",
    "Score above 750 is considered excellent for loan approval.",
    "Score below 650 may result in loan rejection.",
    "Pay EMIs and credit card bills on time to maintain good score.",
    "Keep credit utilization below 30% of credit limit.",
    "",
    "4. INSURANCE",
    "Term life insurance cover should be 10-15 times annual income.",
    "Health insurance minimum Rs 5 lakh cover recommended.",
    "ULIP combines insurance and investment but has high charges.",
    "Pure term plan is cheaper than ULIP for same coverage.",
    "Critical illness rider provides lump sum on diagnosis.",
    "",
    "5. RETIREMENT PLANNING",
    "NPS (National Pension System) offers additional Rs 50,000 deduction.",
    "PPF interest rate is set by government quarterly, currently 7.1%.",
    "EPF contribution is 12% of basic salary by employee and employer.",
    "Start retirement planning before age 30 for best results.",
    "Target retirement corpus = 25 times annual expenses.",
]
for line in content3:
    pdf3.cell(0, 10, line, ln=True)
pdf3.output("data/knowledge_base/personal_finance_guide.pdf")
print("Created: personal_finance_guide.pdf")

print("\nAll 3 PDFs created in data/knowledge_base/")
print("Now run: python rag/ingest.py")