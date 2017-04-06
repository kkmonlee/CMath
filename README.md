# CMath
CMath is an extremely small recursive top-down (descent) parser and a mathematical evaluation engine.

It gives you the ability to evaluate mathematical expressions at runtime without adding more code to your project. Additionally, CMath also supports the standard C mathematical functions and runtime binding of variables.

## Features

- **ANSI C** with no dependencies
- Single source and header file
- Efficient and simple
- Implements [standard operator precedence](http://en.cppreference.com/w/c/language/operator_precedence)
- Uses standard C mathematical functions (`sin`, `sqrt`, `atan`, etc.)
- Ability to bind variables at program evaluation
- Released under GNU GPL v3 license
- Thread-safe, assuming your **`malloc`** is

# Example
```c
#include "cmath.h"

printf("%f\n", cm_interp("5*5", 0)); /* Prints 25 */
```

## Usage
CMath only defines 4 functions:

```c
double cm_interp(const char *expression, int *error);

cm_expr *cm_compile(const char *expression, const cm_variable *variables, int var_count, int *error);

double cm_eval(const cm_expr *expr);

void cm_free(cm_expr *expr);
```

###`cm_interp`
`cm_interp()` takes an expression and immediately returns the result of it. If there is a format or parsing error, `cm_interp()` returns `NaN`.

If the `error` pointer argument is not 0, then `cm_interp()` will set `*error` to the position of the parse error on failure, and set `*error` to 0 on success.

#### Usage
```c
int error;

double a = cm_interp("(5+5)", 0); /* Returns 10. */
double b = cm_interp("(5+5)", &error); /* Returns 10, error is set to 0. */
double c = cm_interp("(5+5", &error); /* Returns NaN, error is set to 4. */
```

### `cm_compile`, `cm_eval`, `cm_free`
```c
cm_expr *cm_compile(const char *expression, const cm_variable *lookup, int lookup_len, int *error);
double cm_eval(const cm_expr *n);
void cm_free(cm_expr *n);
```

`cm_compile()` must be given an expression with unbound variables and a list of variable names and pointers. It will then return a `cm_expr*` which can be evaluated later using `cm_eval()`. On failure, `cm_compile()` will return 0 and optionally set the passed in `*error` to the location of the parse error.

You can also compile expressions without variables by passing `cm_compile()`'s second and third arguments as 0.

A `cm_expr*` must be given to `cm_eval()` from `cm_compile()`; `cm_eval()` will then evaluate the expression using the current variable values. 

At the end, remember to invoke `cm_free()`.

#### Usage
```c
double x, y;
// Variable names and pointers
cm_variables vars[] = {{"x", &x}, {"y", &y}};

int err;
// Compile the expression with variables
cm_expr *expr = cm_compile("sqrt(x^2+y^2)", vars, 2, &err);

if (expr) {
	x = 3; y = 4;
	const double h1 = cm_eval(expr); // Returns 5
	
	x = 5; y = 12;
	const double h2 = cm_eval(expr); // Returns 13
	
	cm_free(expr);
} else {
	printf("Parse error at %d\n", err);
}
```

#### Longer example
Here is an example where an expression is passed from the command line and evaluated. It also does error checking and binds the variables `x` and `y` to *3* and *4*, respectively.

```c
#include "cmath.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
	if (argc < 2) {
		printf("Usage: example \"expression\"\n");
		return 0;
	}

	const char *expression = argv[1];
	printf("Evaluating:\n\t%s\n", expression);

	// Variables are bound at program evaluation
	double x, y;
	cm_variable vars[] = {{"x", &x}, {"y", &y}};

	// Check for errors
	int err;
	cm_expr *n = cm_compile(expression, vars, 2, &err);

	if (n) {
		// It is efficient because variables can now be called as many 
		// times as you want because parsing has already been done
		x = 3; y = 4;
		const double r = cm_eval(n); printf("Result:\n\t%f\n", r);
		cm_free(n);
	} else {
		// Show error
		printf("\t%*s^\nError near here", err-1, "");
	}

	return 0;
}

```

Which produces the output:
```
$	example "sqrt(x^2+y2)"
	Evaluating:
		sqrt(x^2+y2)
			      ^
	Error near here
```

## Speed
CMath is fast compared to C in all regards. 

Here are some example performance numbers taken from a benchmark:

|**Expression** |**`cm_eval` time**| **native C time** | **difference**|
|----------------------|----------------------------|---------------------------|---------------------|
|`sqrt(a^1.5+a^2.5)`|14.478 ms|15.641 ms|8% faster|
|`a+5`|563 ms|765 ms|35% faster|
|`a+(5*2)`|563 ms|765 ms|36% faster|
|`(a+5)*2`|563 ms|1422 ms|153% faster|
|`(1/(a+1)+2/(a+2)+3/(a+3))`|1266 ms | 5516 ms | 336% faster|


## Grammar
CMath uses and parses the following grammar:
```
<list>      =    <expr> {"," <expr>}
<expr>      =    <term> {("+" | "-") <term>}
<term>      =    <factor> {("*" | "/" | "%") <factor>}
<factor>    =    <power> {"^" <power>}
<power>     =    {("-" | "+")} <base>
<base>      =    <constant> | <variable> | <function-0> {"(" ")"} | <function-1> <power> | <function-2> "(" <expr> "," <expr> ")" | "(" <list> ")"
```
- Also, whitespace between tokens are ignored. 

- Valid variable names are any combinations of lower case letters from *a* to *z*. 

- Constants can be integers, decimals, or in scientific notation (e.g., 1e3).

- A leading zero is not required 

## Functions supported
CMath supports addition, subtraction, multiplication, division, exponentiation and modulus with normal operator precedence (one exception being that exponentiation is evaluated left-to-right).

Additionally, following C mathematical functions are also supported:

- `abs` (`fabs`)
- `acos`
- `asin`
- `atan`
- `atan2`
- `ceil`
- `cos
- `cosh`
- `exp`
- `floor`
- `ln` (`log`)
- `log` (`log10`)
- `pow`
- `sin`
- `sinh`
- `sqrt`
- `tan`
- `tanh`

The following constants are also available:

- `pi`
- `e`

---

- All functions/types start with `cm`
