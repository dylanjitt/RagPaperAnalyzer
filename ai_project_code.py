
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats
import uvicorn

# Database simulation
class Database:
    def __init__(self):
        self.data = {}

    def save(self, key, value):
        self.data[key] = value

    def retrieve(self, key):
        return self.data.get(key, None)

# Model for request body
class MathOperation(BaseModel):
    operation: str
    values: list

# Initialize FastAPI and Database
app = FastAPI()
db = Database()

# Define mathematical operations
@app.post("/math/")
def perform_math_operation(op: MathOperation):
    try:
        if op.operation == 'mean':
            result = np.mean(op.values)
        elif op.operation == 'median':
            result = np.median(op.values)
        elif op.operation == 'mode':
            result = stats.mode(op.values).mode[0]
        else:
            raise HTTPException(status_code=400, detail="Invalid operation")
            
        # Save the result to the database
        db.save(op.operation, result)
        return {"operation": op.operation, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to retrieve result
@app.get("/math/{operation}")
def get_result(operation: str):
    result = db.retrieve(operation)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return {"operation": operation, "result": result}

# Run the application
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


### Explanation:
# - **FastAPI** is used to create a web application that includes endpoints for performing mathematical operations.
# - **Pydantic** is employed for request validation through a `MathOperation` model which captures the operation type and values.
# - **Numpy** and **Scipy** provide the computational capabilities to calculate mean, median, and mode.
# - A simple in-memory **Database** class simulates data persistence.
# - The application has two endpoints: one for performing mathematical operations and the other for retrieving previously computed results.
# - This architecture supports maintainability and scalability, with the potential for future enhancements such as connecting to a real database or handling more complex operations.

# This implementation provides a solid foundation for the AI project as described.