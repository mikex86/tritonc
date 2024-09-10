#include "tinyscript.h"
#include <sstream>
#include <stack>
#include <string>
#include <cctype>
#include <stdexcept>

uint64_t applyOperator(uint64_t a, uint64_t b, char op) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/':
            if (b == 0) throw std::runtime_error("Division by zero");
            return a / b;
        default: throw std::invalid_argument("Invalid operator");
    }
}

uint64_t precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/') return 2;
    return 0;
}

void processTopOperator(std::stack<uint64_t>& values, std::stack<char>& ops) {
    uint64_t val2 = values.top();
    values.pop();
    uint64_t val1 = values.top();
    values.pop();
    char op = ops.top();
    ops.pop();
    values.push(applyOperator(val1, val2, op));
}

uint64_t tinyScriptEval(const std::string &expression) {
    std::stack<uint64_t> values;   // Stack to store integer values.
    std::stack<char> ops;     // Stack to store operators.

    for (size_t i = 0; i < expression.length(); ++i) {
        if (isdigit(expression[i])) {
            // Read the number (can have more than one digit)
            int value = 0;
            while (i < expression.length() && isdigit(expression[i])) {
                value = value * 10 + (expression[i] - '0');
                i++;
            }
            values.push(value);
            i--;  // Adjust for the loop increment
        } else if (expression[i] == '(') {
            ops.push(expression[i]);
        } else if (expression[i] == ')') {
            // Solve entire bracketed expression
            while (!ops.empty() && ops.top() != '(') {
                processTopOperator(values, ops);
            }
            ops.pop();  // Remove '('
        } else if (expression[i] == '+' || expression[i] == '-' || expression[i] == '*' || expression[i] == '/') {
            // Handle operator precedence
            while (!ops.empty() && precedence(ops.top()) >= precedence(expression[i])) {
                processTopOperator(values, ops);
            }
            ops.push(expression[i]);
        }
    }

    // Apply remaining operators
    while (!ops.empty()) {
        processTopOperator(values, ops);
    }

    // The result is the last value in the stack
    return values.top();
}
