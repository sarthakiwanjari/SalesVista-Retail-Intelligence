from pathlib import Path
import io
import pandas as pd
from flask import Flask, render_template, jsonify, request, Response
from analysis.sales_analysis import load_data, dashboard_payload, filter_data, _records

BASE = Path(__file__).resolve().parent
app = Flask(__name__, template_folder="templates", static_folder="public/static", static_url_path="/static")

def get_df():
    return load_data(BASE / "data" / "superstore.csv")

@app.route("/")
def index():
    try:
        df=get_df()
        status={"ready":True,"rows":len(df)}
    except Exception as exc:
        status={"ready":False,"error":str(exc),"rows":0}
    return render_template("index.html", dataset_status=status)

@app.route("/api/analytics")
def analytics():
    try:
        df=get_df()
        params={k:request.args.get(k) for k in ("start","end","region","category","segment")}
        payload=dashboard_payload(df,params)
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error":str(exc),"type":type(exc).__name__}), 400

@app.route("/api/export.csv")
def export_csv():
    try:
        df=filter_data(get_df(),{k:request.args.get(k) for k in ("start","end","region","category","segment")})
        out=io.StringIO(); df.to_csv(out,index=False)
        return Response(out.getvalue(),mimetype="text/csv",headers={"Content-Disposition":"attachment; filename=salesvista_filtered_data.csv"})
    except Exception as exc:
        return jsonify({"error":str(exc)}),400

@app.route("/api/health")
def health():
    try: return jsonify({"ok":True,"rows":len(get_df())})
    except Exception as exc: return jsonify({"ok":False,"error":str(exc)}),503

@app.errorhandler(404)
def not_found(_): return jsonify({"error":"Route not found"}),404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
