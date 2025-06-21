import astroid
from pylint.checkers import BaseChecker


class MLCodeSmellChecker(BaseChecker):
    name = 'ml-code-smells'
    priority = -1
    msgs = {
        'W9002': (
            'Hyperparameter not explicitly set in call to "%s" — using default values',
            'implicit-hyperparameter',
            'Important hyperparameters should be set explicitly to avoid silent defaults.',
        ),
    }

    TARGET_FUNCTIONS = {
        'GaussianNB': ['var_smoothing'],
        'train_test_split': ['test_size', 'random_state'],
    }

    def visit_call(self, node):
        func_name = self._get_call_func_name(node)

        if func_name in self.TARGET_FUNCTIONS:
            explicitly_set = {kw.arg for kw in node.keywords if kw.arg}
            expected_hyperparams = set(self.TARGET_FUNCTIONS[func_name])

            missing = expected_hyperparams - explicitly_set
            if missing == expected_hyperparams:
                self.add_message(
                    'implicit-hyperparameter',
                    node=node,
                    args=(func_name,)
                )

    def _get_call_func_name(self, node):
        if isinstance(node.func, astroid.Name):
            return node.func.name
        elif isinstance(node.func, astroid.Attribute):
            return node.func.attrname
        return None


def register(linter):
    linter.register_checker(MLCodeSmellChecker(linter))
