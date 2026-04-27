import marimo

__generated_with = "0.23.3"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell
def _(mo):
    btn = mo.ui.run_button(label="Click me")
    btn
    return (btn,)

@app.cell
def _(btn, mo):
    mo.stop(not btn.value, "Not clicked yet")
    result = 42
    mo.md(f"Result: {result}")
    return (result,)

@app.cell
def _(result, mo):
    mo.md(f"Downstream sees: {result}")
    return

if __name__ == "__main__":
    app.run()
