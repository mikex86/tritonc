#include "templating.h"
#include "tinyscript.h"

#include <sstream>
#include <cassert>

std::string applyTemplate(const std::string &codeTemplate,
                          const std::unordered_map<std::string, int> &autotuneValues) {

    // apply template
    std::stringstream templated;
    for (size_t i = 0; i < codeTemplate.size(); i++) {
        if (codeTemplate[i] == '$') {
            i++;
            assert(codeTemplate[i] == '{');
            i++;
            std::string tempalte_expression{};
            size_t paren_count = 1;
            while (true) {
                if (codeTemplate[i] == '{') {
                    paren_count++;
                }
                if (codeTemplate[i] == '}') {
                    paren_count--;
                    if (paren_count == 0) {
                        break;
                    }
                }
                tempalte_expression += codeTemplate[i];
                i++;
            }
            i++;
            if (tempalte_expression.find('$') != std::string::npos) {
                tempalte_expression = applyTemplate(tempalte_expression, autotuneValues);
            }
            if (autotuneValues.find(tempalte_expression) != autotuneValues.end()) {
                templated << std::to_string(autotuneValues.find(tempalte_expression)->second);
            } else {
                templated << std::to_string(tinyScriptEval(tempalte_expression));
            }
        }
        templated << codeTemplate[i];
    }
    return templated.str();
}