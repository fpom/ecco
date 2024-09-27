import ast

class Node2SDD (ast.NodeVisitor) :
    def __init__ (self, var2sdd) :
        self.sdd = var2sdd
    def generic_visit (self, node) :
        raise ValueError("invalid state expression (%r not supported)"
                         % node.__class__.__name__)
    def visit_Expression (self, node) :
        return self.visit(node.body)
    def visit_Name (self, node) :
        return self.sdd(node.id)
    def visit_BinOp (self, node) :
        if isinstance(node.op, ast.BitOr) :
            return self.visit(node.left) | self.visit(node.right)
        elif isinstance(node.op, ast.BitAnd) :
            return self.visit(node.left) & self.visit(node.right)
        elif isinstance(node.op, ast.BitXor) :
            return ((self.visit(node.left) | self.visit(node.right))
                    & (self.sdd("*") - (self.visit(node.left) & self.visit(node.right))))
        else :
            raise ValueError("invalid state expression (%r not supported)"
                             % node.op.__class__.__name__)
    def visit_UnaryOp (self, node) :
        if isinstance(node.op, ast.Invert) :
            return self.sdd("*") - self.visit(node.operand)
        else :
            raise ValueError("invalid state expression (%r not supported)"
                             % node.op.__class__.__name__)

def expr2sdd (expr, var2sdd) :
    node2SDD = Node2SDD(var2sdd)
    tree = ast.parse(expr, mode="eval")
    return node2SDD.visit(tree)
