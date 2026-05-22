import sys
import re
import asyncio
import threading
import math
import urllib.request
import http.server
import inspect

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GU import *
import pygame.sndarray
import numpy

# --------------- Tokens ---------------
KEYWORDS = {
    'meow': 'MEOW', 'purr': 'PURR', 'purr_else': 'PURR_ELSE',
    'scratch': 'SCRATCH', 'whiskers': 'WHISKERS', 'catnap': 'CATNAP',
    'hiss': 'HISS', 'tail': 'TAIL', 'claw': 'CLAW',
    'async_meow': 'ASYNC_MEOW', 'await': 'AWAIT',
    'in': 'IN', 'and': 'AND', 'or': 'OR', 'not': 'NOT',
    'true': 'TRUE', 'false': 'FALSE', 'none': 'NONE'
}

class Token:
    def __init__(self, typ, value, line, col):
        self.typ = typ
        self.value = value
        self.line = line
        self.col = col
    def __repr__(self): return f"Token({self.typ}, {self.value})"

def tokenize(code):
    spec = [
        ('NUMBER', r'\d+(\.\d*)?'),
        ('STRING', r'"[^"]*"|\'[^\']*\''),
        ('LBRACE', r'\{'), ('RBRACE', r'\}'),
        ('LPAREN', r'\('), ('RPAREN', r'\)'),
        ('LBRACKET', r'\['), ('RBRACKET', r'\]'),
        ('COMMA', r','), ('DOT', r'\.'),
        ('OP', r'==|!=|<=|>=|[+\-*/%=<>]'),
        ('ID', r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('NEWLINE', r'\n'), ('SKIP', r'[ \t]+'),
        ('MISMATCH', r'.')
    ]
    tok_regex = '|'.join(f'(?P<{p}>{r})' for p, r in spec)
    line = 1
    line_start = 0
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'NEWLINE':
            line += 1
            line_start = mo.end()
            continue
        elif kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise SyntaxError(f"Unexpected character {value!r} at line {line}")
        else:
            col = mo.start() - line_start
            if kind == 'ID' and value in KEYWORDS:
                kind = KEYWORDS[value]
            yield Token(kind, value, line, col)

# --------------- AST ---------------
class ASTNode: pass

class Program(ASTNode):
    def __init__(self, stmts):
        self.stmts = stmts

class Assign(ASTNode):
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

class Print(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class If(ASTNode):
    def __init__(self, cond, then_body, else_body):
        self.cond = cond
        self.then_body = then_body
        self.else_body = else_body

class For(ASTNode):
    def __init__(self, var, iterable, body):
        self.var = var
        self.iterable = iterable
        self.body = body

class While(ASTNode):
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

class FuncDef(ASTNode):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class Call(ASTNode):
    def __init__(self, func, args):
        self.func = func
        self.args = args

class Return(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class RaiseStmt(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class Sleep(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class AsyncBlock(ASTNode):
    def __init__(self, body):
        self.body = body

class AwaitStmt(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class ExprStatement(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class BinOp(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class UnaryOp(ASTNode):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

class Literal(ASTNode):
    def __init__(self, value):
        self.value = value

class Name(ASTNode):
    def __init__(self, name):
        self.name = name

class ListLiteral(ASTNode):
    def __init__(self, elements):
        self.elements = elements

class Index(ASTNode):
    def __init__(self, obj, index):
        self.obj = obj
        self.index = index

class Attribute(ASTNode):
    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr

# --------------- Parser ---------------
class Parser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.pos = 0
    def peek(self):
        if self.pos < len(self.tokens): return self.tokens[self.pos]
        return Token('EOF', '', -1, -1)
    def advance(self):
        t = self.peek()
        if t.typ != 'EOF': self.pos += 1
        return t
    def expect(self, typ, val=None):
        t = self.peek()
        if t.typ != typ or (val is not None and t.value != val):
            raise SyntaxError(f"Expected {typ}{' '+val if val else ''}, got {t}")
        return self.advance()
    def match(self, typ, val=None):
        t = self.peek()
        if t.typ == typ and (val is None or t.value == val):
            return self.advance()
        return None
    def parse(self):
        stmts = []
        while self.peek().typ != 'EOF':
            stmts.append(self.parse_stmt())
        return Program(stmts)
    def parse_stmt(self):
        t = self.peek()
        if t.typ == 'MEOW':
            self.advance()
            return Print(self.parse_expr())
        elif t.typ == 'PURR':
            self.advance()
            cond = self.parse_expr()
            self.expect('LBRACE')
            then = self.parse_block()
            else_ = None
            if self.peek().typ == 'PURR_ELSE':
                self.advance(); self.expect('LBRACE'); else_ = self.parse_block()
            return If(cond, then, else_)
        elif t.typ == 'SCRATCH':
            self.advance()
            var = self.expect('ID').value
            self.expect('IN')
            iter = self.parse_expr()
            self.expect('LBRACE')
            body = self.parse_block()
            return For(var, iter, body)
        elif t.typ == 'WHISKERS':
            self.advance()
            cond = self.parse_expr()
            self.expect('LBRACE')
            body = self.parse_block()
            return While(cond, body)
        elif t.typ == 'CATNAP':
            self.advance()
            return Sleep(self.parse_expr())
        elif t.typ == 'HISS':
            self.advance()
            return RaiseStmt(self.parse_expr())
        elif t.typ == 'TAIL':
            self.advance()
            e = self.parse_expr() if self.peek().typ not in ('RBRACE','EOF','NEWLINE') else None
            return Return(e)
        elif t.typ == 'CLAW':
            self.advance()
            name = self.expect('ID').value
            self.expect('LPAREN')
            params = []
            if self.peek().typ != 'RPAREN':
                params.append(self.expect('ID').value)
                while self.match('COMMA'):
                    params.append(self.expect('ID').value)
            self.expect('RPAREN')
            self.expect('LBRACE')
            body = self.parse_block()
            return FuncDef(name, params, body)
        elif t.typ == 'ASYNC_MEOW':
            self.advance(); self.expect('LBRACE')
            return AsyncBlock(self.parse_block())
        elif t.typ == 'AWAIT':
            self.advance()
            return AwaitStmt(self.parse_expr())
        elif self.peek().typ == 'ID' and len(self.tokens)>self.pos+1 and self.tokens[self.pos+1].value == '=':
            var = self.advance().value
            self.expect('OP', '=')
            return Assign(var, self.parse_expr())
        else:
            return ExprStatement(self.parse_expr())
    def parse_block(self):
        stmts = []
        while self.peek().typ not in ('RBRACE','EOF'):
            stmts.append(self.parse_stmt())
        self.expect('RBRACE')
        return stmts
    def parse_expr(self): return self.parse_or()
    def parse_or(self):
        l = self.parse_and()
        while self.match('OR'): l = BinOp(l, 'or', self.parse_and())
        return l
    def parse_and(self):
        l = self.parse_not()
        while self.match('AND'): l = BinOp(l, 'and', self.parse_not())
        return l
    def parse_not(self):
        if self.match('NOT'): return UnaryOp('not', self.parse_not())
        return self.parse_comp()
    def parse_comp(self):
        l = self.parse_arith()
        if self.peek().typ == 'OP' and self.peek().value in ('==','!=','<','>','<=','>='):
            op = self.advance().value
            return BinOp(l, op, self.parse_arith())
        return l
    def parse_arith(self):
        l = self.parse_term()
        while self.peek().typ == 'OP' and self.peek().value in ('+','-'):
            op = self.advance().value
            l = BinOp(l, op, self.parse_term())
        return l
    def parse_term(self):
        l = self.parse_unary()
        while self.peek().typ == 'OP' and self.peek().value in ('*','/','%'):
            op = self.advance().value
            l = BinOp(l, op, self.parse_unary())
        return l
    def parse_unary(self):
        if self.peek().typ == 'OP' and self.peek().value in ('+','-'):
            op = self.advance().value
            return UnaryOp(op, self.parse_unary())
        return self.parse_postfix()
    def parse_postfix(self):
        e = self.parse_atom()
        while True:
            if self.match('LPAREN'):
                args = []
                if self.peek().typ != 'RPAREN':
                    args.append(self.parse_expr())
                    while self.match('COMMA'): args.append(self.parse_expr())
                self.expect('RPAREN')
                e = Call(e, args)
            elif self.match('LBRACKET'):
                ix = self.parse_expr()
                self.expect('RBRACKET')
                e = Index(e, ix)
            elif self.match('DOT'):
                attr = self.expect('ID').value
                e = Attribute(e, attr)
            else: break
        return e
    def parse_atom(self):
        t = self.peek()
        if t.typ == 'NUMBER':
            self.advance()
            return Literal(float(t.value) if '.' in t.value else int(t.value))
        elif t.typ == 'STRING':
            self.advance()
            return Literal(t.value[1:-1])
        elif t.typ == 'TRUE': self.advance(); return Literal(True)
        elif t.typ == 'FALSE': self.advance(); return Literal(False)
        elif t.typ == 'NONE': self.advance(); return Literal(None)
        elif t.typ == 'ID': return Name(self.advance().value)
        elif t.typ == 'LPAREN':
            self.advance(); e = self.parse_expr(); self.expect('RPAREN')
            return e
        elif t.typ == 'LBRACKET':
            self.advance()
            items = []
            if self.peek().typ != 'RBRACKET':
                items.append(self.parse_expr())
                while self.match('COMMA'): items.append(self.parse_expr())
            self.expect('RBRACKET')
            return ListLiteral(items)
        else:
            raise SyntaxError(f"Unexpected token {t}")

# --------------- Runtime ---------------
class ReturnException(Exception):
    def __init__(self, value): self.value = value
class HissException(Exception): pass

class Environment:
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent
    def get(self, name):
        if name in self.vars: return self.vars[name]
        if self.parent: return self.parent.get(name)
        raise Exception(f"Name '{name}' is not defined")
    def set(self, name, value):
        if name in self.vars:
            self.vars[name] = value; return
        if self.parent and name in self.parent.vars:
            self.parent.set(name, value); return
        self.vars[name] = value
    def define(self, name, value): self.vars[name] = value

class ILFunction:
    def __init__(self, name, params, body, closure):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure
    async def call(self, interp, args):
        if len(args) != len(self.params):
            raise Exception(f"Function {self.name} expects {len(self.params)} args, got {len(args)}")
        env = Environment(self.closure)
        for p, a in zip(self.params, args):
            env.define(p, a)
        try:
            for stmt in self.body:
                await interp.eval_stmt(stmt, env)
        except ReturnException as e:
            return e.value
        return None

class Runtime:
    def __init__(self):
        self.screen = None
        self.clock = None
        self.window_open = False
        self.camera_control = False
        self.camera_rot_x = 0
        self.camera_rot_y = 0
        self.cam_distance = 5
        self.last_mouse_pos = (0,0)
        self.cat_position = (0,0,0)
        self.pet_anim = 0
        self.on_click_callback = None
        self.meow_sound = None
        self.pygame_task = None

# --------------- Interpreter ---------------
class ILoveCatsInterpreter:
    def __init__(self):
        self.global_env = Environment()
        self.runtime = Runtime()
        self.loop = None
        self._init_builtins()
    def _init_builtins(self):
        for name, func in {
            'window': self._window,
            'draw_cat': self._draw_cat,
            'draw_3d_cat': self._draw_3d_cat,
            'pet_cat': self._pet_cat,
            'on_click': self._on_click,
            'camera_control': self._camera_control,
            'wait_for_close': self._wait_for_close,
            'http_server': self._http_server,
            'http_get': self._http_get,
            'websocket_connect': self._websocket_connect,
            'cat_read': self._cat_read,
            'cat_write': self._cat_write,
            'catnap': self._catnap,
            'range': self._range,
            'len': self._len,
            'str': self._str,
            'int': self._int,
            'float': self._float,
            'print': self._print,
        }.items():
            self.global_env.define(name, func)

    # --- statement evaluation ---
    async def eval_stmt(self, stmt, env):
        if isinstance(stmt, Print):
            val = await self.eval_expr(stmt.expr, env)
            print(val)
        elif isinstance(stmt, Assign):
            env.set(stmt.name, await self.eval_expr(stmt.expr, env))
        elif isinstance(stmt, If):
            if await self.eval_expr(stmt.cond, env):
                for s in stmt.then_body: await self.eval_stmt(s, env)
            elif stmt.else_body:
                for s in stmt.else_body: await self.eval_stmt(s, env)
        elif isinstance(stmt, For):
            it = await self.eval_expr(stmt.iterable, env)
            if not hasattr(it, '__iter__'): raise Exception("Неитерируемый объект в scratch")
            loop_env = Environment(env)
            for item in it:
                loop_env.define(stmt.var, item)
                for s in stmt.body: await self.eval_stmt(s, loop_env)
        elif isinstance(stmt, While):
            while await self.eval_expr(stmt.cond, env):
                for s in stmt.body: await self.eval_stmt(s, env)
        elif isinstance(stmt, FuncDef):
            env.define(stmt.name, ILFunction(stmt.name, stmt.params, stmt.body, env))
        elif isinstance(stmt, Return):
            val = await self.eval_expr(stmt.expr, env) if stmt.expr else None
            raise ReturnException(val)
        elif isinstance(stmt, RaiseStmt):
            raise HissException(str(await self.eval_expr(stmt.expr, env)))
        elif isinstance(stmt, Sleep):
            await asyncio.sleep(float(await self.eval_expr(stmt.expr, env)))
        elif isinstance(stmt, AsyncBlock):
            async def block():
                enc = Environment(env)
                for s in stmt.body: await self.eval_stmt(s, enc)
            asyncio.create_task(block())
        elif isinstance(stmt, AwaitStmt):
            await self.eval_expr(stmt.expr, env)
        elif isinstance(stmt, ExprStatement):
            await self.eval_expr(stmt.expr, env)
        else:
            raise Exception(f"Unknown statement {type(stmt)}")

    # --- expression evaluation ---
    async def eval_expr(self, expr, env):
        if isinstance(expr, Literal): return expr.value
        elif isinstance(expr, Name): return env.get(expr.name)
        elif isinstance(expr, BinOp):
            l = await self.eval_expr(expr.left, env)
            r = await self.eval_expr(expr.right, env)
            op = expr.op
            if op == '+': return l + r
            if op == '-': return l - r
            if op == '*': return l * r
            if op == '/': return l / r
            if op == '%': return l % r
            if op == '==': return l == r
            if op == '!=': return l != r
            if op == '<': return l < r
            if op == '>': return l > r
            if op == '<=': return l <= r
            if op == '>=': return l >= r
            if op == 'and': return l and r
            if op == 'or': return l or r
        elif isinstance(expr, UnaryOp):
            opnd = await self.eval_expr(expr.operand, env)
            if expr.op == '-': return -opnd
            if expr.op == '+': return +opnd
            if expr.op == 'not': return not opnd
        elif isinstance(expr, Call):
            func = await self.eval_expr(expr.func, env)
            args = [await self.eval_expr(a, env) for a in expr.args]
            if isinstance(func, ILFunction):
                return await func.call(self, args)
            elif callable(func):
                if hasattr(func, '__self__') and func.__self__ is not None:
                    if inspect.iscoroutinefunction(func):
                        return await func(*args)
                    else:
                        return func(*args)
                else:
                    if inspect.iscoroutinefunction(func):
                        return await func(self, args)
                    else:
                        return func(self, args)
            raise Exception(f"'{expr.func}' не вызываемый объект")
        elif isinstance(expr, ListLiteral):
            return [await self.eval_expr(e, env) for e in expr.elements]
        elif isinstance(expr, Index):
            obj = await self.eval_expr(expr.obj, env)
            ix = await self.eval_expr(expr.index, env)
            return obj[ix]
        elif isinstance(expr, Attribute):
            obj = await self.eval_expr(expr.obj, env)
            return getattr(obj, expr.attr)
        raise Exception(f"Unknown expression {type(expr)}")

    # --------------- builtins ---------------
    async def _window(self, args):
        title, w, h = args
        pygame.init()
        pygame.mixer.init()
        pygame.display.set_caption(str(title))
        display = pygame.display.set_mode((int(w), int(h)), DOUBLEBUF | OPENGL)
        glEnable(GL_DEPTH_TEST)
        self.runtime.screen = display
        self.runtime.clock = pygame.time.Clock()
        self.runtime.window_open = True
        self.runtime.pygame_task = asyncio.create_task(self._pygame_loop())
    async def _pygame_loop(self):
        while self.runtime.window_open:
            for ev in pygame.event.get():
                if ev.type == QUIT:
                    self.runtime.window_open = False
                elif ev.type == MOUSEBUTTONDOWN:
                    if self.runtime.on_click_callback:
                        asyncio.create_task(self._invoke_click())
                elif ev.type == MOUSEMOTION and self.runtime.camera_control:
                    if pygame.mouse.get_pressed()[0]:
                        self.runtime.camera_rot_y += ev.rel[0] * 0.005
                        self.runtime.camera_rot_x += ev.rel[1] * 0.005
            self._render()
            pygame.display.flip()
            self.runtime.clock.tick(60)
            await asyncio.sleep(0)
    async def _invoke_click(self):
        if self.runtime.on_click_callback:
            await self.runtime.on_click_callback.call(self, [])
    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        aspect = self.runtime.screen.get_width() / max(1, self.runtime.screen.get_height())
        gluPerspective(45, aspect, 0.1, 50.0)
        glTranslatef(0, 0, -self.runtime.cam_distance)
        glRotatef(math.degrees(self.runtime.camera_rot_x), 1, 0, 0)
        glRotatef(math.degrees(self.runtime.camera_rot_y), 0, 1, 0)
        self._draw_cat(*self.runtime.cat_position)
    def _draw_cat(self, x, y, z):
        glPushMatrix()
        glTranslatef(x, y, z)
        if self.runtime.pet_anim > 0:
            scale = 1.0 + 0.1 * math.sin(pygame.time.get_ticks() * 0.01)
            self.runtime.pet_anim -= 1
        else:
            scale = 1.0
        glScalef(scale, scale, scale)
        glColor3f(1.0, 0.6, 0.0)  # оранжевый
        self._sphere(1.0)
        glColor3f(1.0, 0.75, 0.8)  # ушки
        for (sx, sy, angle) in [(-0.5, 0.8, -20), (0.5, 0.8, 20)]:
            glPushMatrix()
            glTranslatef(sx, sy, 0)
            glRotatef(angle, 0, 0, 1)
            glRotatef(90, 1, 0, 0)
            self._cone(0.5, 0.5)
            glPopMatrix()
        glColor3f(0, 0, 0)  # глазки
        for (ex, ey, ez) in [(-0.3, 0.3, 0.8), (0.3, 0.3, 0.8)]:
            glPushMatrix(); glTranslatef(ex, ey, ez); self._sphere(0.15); glPopMatrix()
        glColor3f(1, 0.5, 0.5)  # нос
        glPushMatrix(); glTranslatef(0, 0.1, 1.0); self._sphere(0.1); glPopMatrix()
        glPopMatrix()
    def _sphere(self, r, slices=16, stacks=16):
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + i / stacks)
            z0 = math.sin(lat0); zr0 = math.cos(lat0)
            lat1 = math.pi * (-0.5 + (i+1) / stacks)
            z1 = math.sin(lat1); zr1 = math.cos(lat1)
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(slices+1):
                lng = 2 * math.pi * j / slices
                x = math.cos(lng); y = math.sin(lng)
                glVertex3f(x*zr0*r, y*zr0*r, z0*r)
                glVertex3f(x*zr1*r, y*zr1*r, z1*r)
            glEnd()
    def _cone(self, base_r, h, slices=16):
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, h)
        for i in range(slices+1):
            a = 2 * math.pi * i / slices
            glVertex3f(math.cos(a)*base_r, math.sin(a)*base_r, 0)
        glEnd()
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        for i in range(slices+1):
            a = 2 * math.pi * i / slices
            glVertex3f(math.cos(a)*base_r, math.sin(a)*base_r, 0)
        glEnd()
    async def _draw_cat(self, args): pass
    async def _draw_3d_cat(self, args):
        self.runtime.cat_position = (float(args[0]), float(args[1]), float(args[2]))
    async def _pet_cat(self, args):
        self.runtime.pet_anim = 30
        self._play_meow()
    def _play_meow(self):
        if not self.runtime.meow_sound:
            sr = 22050; dur = 0.4; freq = 700
            t = numpy.arange(int(sr*dur))
            wave = numpy.sin(2*math.pi*freq*t/sr) * numpy.exp(-t/(sr*0.1))
            wave = numpy.int16(wave * 32767)
            self.runtime.meow_sound = pygame.sndarray.make_sound(wave)
        self.runtime.meow_sound.play()
    async def _on_click(self, args):
        self.runtime.on_click_callback = args[0]
    async def _camera_control(self, args):
        self.runtime.camera_control = bool(args[0])
    async def _wait_for_close(self, args):
        while self.runtime.window_open:
            await asyncio.sleep(0.1)
    async def _http_server(self, args):
        port, handler_func = args
        if self.loop is None: raise Exception("Нет цикла событий")
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                async def handle():
                    return await handler_func.call(self.server.interp, [self.path])
                future = asyncio.run_coroutine_threadsafe(handle(), self.server.loop)
                try:
                    result = future.result(timeout=5)
                    self.send_response(200); self.end_headers()
                    self.wfile.write(str(result).encode())
                except:
                    self.send_response(500); self.end_headers()
        server = http.server.HTTPServer(('', int(port)), Handler)
        server.interp = self
        server.loop = self.loop
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Сервер запущен на порту {port}")
    async def _http_get(self, args):
        url = str(args[0])
        def fetch():
            with urllib.request.urlopen(url) as f: return f.read().decode()
        return await asyncio.to_thread(fetch)
    async def _websocket_connect(self, args):
        url, on_msg = args
        import websockets
        async def listen():
            try:
                async with websockets.connect(str(url)) as ws:
                    while True:
                        msg = await ws.recv()
                        await on_msg.call(self, [msg])
            except Exception as e: print(f"WebSocket error: {e}")
        asyncio.create_task(listen())
    async def _cat_read(self, args):
        with open(str(args[0]), 'r') as f: return f.read()
    async def _cat_write(self, args):
        with open(str(args[0]), 'w') as f: f.write(str(args[1]))

    # --------------- other builtins ---------------
    async def _catnap(self, args):
        await asyncio.sleep(float(args[0]))
    async def _range(self, args):
        if len(args)==1: return range(int(args[0]))
        if len(args)==2: return range(int(args[0]), int(args[1]))
        raise Exception("range: 1 или 2 аргумента")
    async def _len(self, args): return len(args[0])
    async def _str(self, args): return str(args[0])
    async def _int(self, args): return int(args[0])
    async def _float(self, args): return float(args[0])
    async def _print(self, args): print(*args)

# --------------- Entry ---------------
def main():
    if len(sys.argv) < 2:
        print("Usage: ilovecats <script.ilc>")
        return
    with open(sys.argv[1], 'r') as f:
        code = f.read()
    tokens = tokenize(code)
    try:
        ast = Parser(tokens).parse()
    except SyntaxError as e:
        print(f"Синтаксическая ошибка: {e}")
        return
    interp = ILoveCatsInterpreter()
    async def run():
        interp.loop = asyncio.get_running_loop()
        for stmt in ast.stmts:
            await interp.eval_stmt(stmt, interp.global_env)
    try:
        asyncio.run(run())
    except HissException as e:
        print(f"Hiss: {e}")
    except Exception as e:
        print(f"Ошибка выполнения: {e}")
    finally:
        if interp.runtime.window_open:
            interp.runtime.window_open = False

if __name__ == '__main__':
    main()