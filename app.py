import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Transaction Monitor', layout='wide')

@st.cache_resource
def load_model():
    bundle = joblib.load('model.joblib')
    return bundle['model'], np.asarray(bundle['history_scores']), bundle['features']

try:
    model, history_scores, features = load_model()
    if len(history_scores) == 0 or not np.isfinite(history_scores).all() or not np.issubdtype(history_scores.dtype, np.number):
        raise ValueError('Historical scores are missing or invalid.')
except Exception:
    st.error('Monitoring unavailable: the trained model or historical scores could not be loaded. Contact the app owner; do not treat transactions as cleared.')
    st.stop()

# DEMO ASSUMPTIONS: verify with the source data dictionary before production.
# 999 represents missing ONLY for transactions_last_hour, not every field.
PLACEHOLDERS = {'transactions_last_hour': [999]}
# Count refers to PREVIOUS transactions in the trailing 60 minutes (not current).
# Exactly 60 minutes ago is excluded from this demonstration window.
WINDOW_MINUTES = 60
FIELDS = ['amount', 'customer_typical_amount', 'transactions_last_hour', 'minutes_since_previous', 'account_age_days']
LABELS = ['Amount', 'Customer typical amount', 'Transactions last hour', 'Minutes since previous transaction', 'Account age']

def validate(values):
    errors = []
    for field, label in zip(FIELDS, LABELS):
        value = values.get(field)
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            errors.append(f'{label}: missing or not numeric.')
            continue
        if not np.isfinite(number):
            errors.append(f'{label}: missing or not finite.')
        elif number in PLACEHOLDERS.get(field, []):
            errors.append(f'{label}: {number:g} is a documented missing-value code in this demo.')
        elif number < 0 or (field in FIELDS[:2] and number == 0):
            errors.append(f'{label}: must be positive.' if field in FIELDS[:2] else f'{label}: cannot be negative.')
        elif field in [FIELDS[2], FIELDS[4]] and not number.is_integer():
            errors.append(f'{label}: must be a whole number.')
    # Cross-field consistency: count of PREVIOUS transactions in the last hour.
    # If the most recent previous transaction is within the window, count >= 1.
    # If it is outside the window, count must be 0.
    # Skip consistency rules when either field is already invalid.
    if not errors:
        gap = float(values['minutes_since_previous'])
        count = float(values['transactions_last_hour'])
        if gap < WINDOW_MINUTES and count == 0:
            errors.append('Inconsistent history: previous transaction was within 60 minutes, but the last-hour count is 0. Check timestamps and count in the source system.')
        elif gap >= WINDOW_MINUTES and count > 0:
            errors.append('Inconsistent history: previous transaction was 60 or more minutes ago, but the last-hour count is above 0. Check timestamps and count in the source system.')
    return errors

def score(values):
    amount, typical, last_hour, gap, age = [float(values[k]) for k in FIELDS]
    row = pd.DataFrame([{
        'log_amount': np.log1p(amount), 'amount_ratio': amount / typical,
        'transactions_last_hour': last_hour, 'log_gap': np.log1p(gap), 'log_age': np.log1p(age)
    }])[features]
    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError('Feature values are not finite.')
    result = float(-model.score_samples(row)[0])
    if not np.isfinite(result):
        raise ValueError('The model returned an invalid score.')
    return result

def show_result(s, cutoff, prefix=''):
    if s > cutoff:
        st.warning(f'{prefix}NEEDS HUMAN REVIEW (score {s:.3f}, cutoff {cutoff:.3f}). This is unusual activity, not a fraud probability or confirmed fraud. Check transaction details and supporting evidence before taking action.')
    else:
        st.info(f'{prefix}NOT FLAGGED BY MODEL (score {s:.3f}, cutoff {cutoff:.3f}). This does not verify that the transaction is legitimate. Follow normal controls and investigate other warning signs.')

st.title('Transaction Monitor')
st.write('Isolation Forest identifies unusual transactions for human review; it does not determine fraud.')
st.caption('Demo assumptions (not verified against the source data dictionary): 999 means missing ONLY for Transactions last hour; this count includes previous transactions within the past 60 minutes, excluding the current transaction and the exact 60-minute boundary. Other fields may legitimately equal 999. Confirm these rules before production use.')
pct = st.sidebar.slider('Historical share above cutoff (%)', 1, 10, 4)
cutoff = np.percentile(history_scores, 100 - pct)
st.sidebar.write(f'Score cutoff: **{cutoff:.3f}**')
st.sidebar.caption('Historical threshold, not a guarantee of current review volume or a fraud probability.')

tab1, tab2 = st.tabs(['Check one transaction', 'Live stream'])
with tab1:
    c1, c2 = st.columns(2)
    amount = c1.number_input('Amount ($)', min_value=0.0, value=60.0)
    typical = c1.number_input("Customer's typical amount ($)", min_value=0.0, value=50.0)
    last_hour = c1.number_input('Transactions in the last hour', min_value=0, value=1)
    gap = c2.number_input('Minutes since previous transaction', min_value=0.0, value=30.0)
    age = c2.number_input('Account age (days)', min_value=0, value=700)
    if st.button('Score this transaction'):
        values = dict(zip(FIELDS, [amount, typical, last_hour, gap, age]))
        errors = validate(values)
        if errors:
            st.error('NOT SCORED — data issue. Correct the source record and try again: ' + '; '.join(errors))
        else:
            try:
                s = score(values)
                show_result(s, cutoff)
                share = (history_scores < s).mean() * 100
                st.caption(f'More unusual than {share:.0f}% of historical transactions. This percentile is not a fraud probability.')
            except Exception:
                st.error('NOT SCORED — model could not process this record. Verify the source data and contact the app owner; do not treat it as cleared.')

with tab2:
    st.write('Scores synthetic transactions from new_transactions.csv, one at a time.')
    speed = st.select_slider('Speed', options=['Slow', 'Medium', 'Fast'], value='Medium')
    pause = {'Slow': 0.5, 'Medium': 0.15, 'Fast': 0.02}[speed]
    if st.button('Start the stream'):
        try:
            stream = pd.read_csv('new_transactions.csv')
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            st.error('STREAM NOT STARTED — transaction CSV is missing or unreadable. Ask the app owner to provide a valid new_transactions.csv and retry. No records were cleared.')
            st.stop()
        if stream.empty:
            st.error('STREAM NOT STARTED — CSV has no transactions. Provide a nonempty file and retry; no records were cleared.')
            st.stop()
        if not stream.columns.is_unique:
            st.error('STREAM NOT STARTED — duplicate CSV column names. Correct the file and retry; no records were cleared.')
            st.stop()
        required = FIELDS + ['transaction_id']
        missing = [c for c in required if c not in stream.columns]
        if missing:
            st.error('STREAM NOT STARTED — missing required columns: ' + ', '.join(missing) + '. Correct the CSV and retry; no records were cleared.')
            st.stop()
        if stream['transaction_id'].isna().any() or stream['transaction_id'].astype(str).str.strip().eq('').any() or stream['transaction_id'].duplicated().any():
            st.error('STREAM NOT STARTED — transaction IDs are missing or duplicated. Fix the IDs and retry so every case can be traced; no records were cleared.')
            st.stop()
        flagged, unscored, checked = [], [], 0
        box, counts, table = st.empty(), st.empty(), st.empty()
        for i, r in stream.iterrows():
            values = r.to_dict()
            errors = validate(values)
            if errors:
                unscored.append({'id': r['transaction_id'], 'reason': '; '.join(errors)})
                box.error(f"{r['transaction_id']}: NOT SCORED — data issue. Correct source data and resubmit for review. " + '; '.join(errors))
            else:
                try:
                    s = score(values)
                    checked += 1
                    if s > cutoff:
                        flagged.append({'id': r['transaction_id'], 'amount': r['amount'], 'score': round(s, 3)})
                    with box.container():
                        show_result(s, cutoff, prefix=f"{r['transaction_id']}: ")
                except Exception:
                    unscored.append({'id': r['transaction_id'], 'reason': 'Model scoring error'})
                    box.error(f"{r['transaction_id']}: NOT SCORED — model error. Escalate for manual review and contact the app owner.")
            counts.write(f'Processed {i + 1} of {len(stream)} | Scored: {checked} | Flagged: {len(flagged)} | Not scored: {len(unscored)}')
            if flagged:
                table.dataframe(pd.DataFrame(flagged), hide_index=True)
            time.sleep(pause)
        if unscored:
            st.error(f'{len(unscored)} transaction(s) were NOT scored. Correct the data and resubmit; escalate unresolved cases for manual review.')
            st.dataframe(pd.DataFrame(unscored), hide_index=True)
        st.info(f'Stream finished: {checked} scored, {len(flagged)} flagged, {len(unscored)} unscored. A model flag is not a fraud determination.')
        if 'needs_review' in stream.columns and not unscored:
            st.caption('Synthetic demonstration labels are available for separate evaluation; they are not shown as confirmed fraud decisions.')
