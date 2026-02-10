def drift_decision(ks_score, threshold=0.2):
    if ks_score < threshold:
        return "Safe"
    elif ks_score < threshold * 2:
        return "Monitor Closely"
    else:
        return "Retrain Required"
        
def drift_decision(ks_score, threshold=0.2):
    if ks_score > threshold:
        return "DRIFT DETECTED"
    return "NO DRIFT"

def retrain_model(model, X_new, y_new):
    model.fit(X_new, y_new)
    return model
