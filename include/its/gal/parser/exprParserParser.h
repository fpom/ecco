/** \file
 *  This C header file was generated by $ANTLR version 3.4
 *
 *     -  From the grammar source file : exprParser.g
 *     -                            On : 2024-08-26 14:56:35
 *     -                for the parser : exprParserParserParser
 *
 * Editing it, at least manually, is not wise.
 *
 * C language generator and runtime by Jim Idle, jimi|hereisanat|idle|dotgoeshere|ws.
 *
 *
 * The parser 
exprParserParser

has the callable functions (rules) shown below,
 * which will invoke the code for the associated rule in the source grammar
 * assuming that the input stream is pointing to a token/text stream that could begin
 * this rule.
 *
 * For instance if you call the first (topmost) rule in a parser grammar, you will
 * get the results of a full parse, but calling a rule half way through the grammar will
 * allow you to pass part of a full token stream to the parser, such as for syntax checking
 * in editors and so on.
 *
 * The parser entry points are called indirectly (by function pointer to function) via
 * a parser context typedef pexprParserParser, which is returned from a call to exprParserParserNew().
 *
 * The methods in pexprParserParser are  as follows:
 *
 *  - 
 void
      pexprParserParser->setGAL(pexprParserParser)
 *  - 
 void
      pexprParserParser->setModel(pexprParserParser)
 *  - 
 void
      pexprParserParser->specification(pexprParserParser)
 *  - 
 its::Composite *
      pexprParserParser->composite(pexprParserParser)
 *  - 
 its::GAL*
      pexprParserParser->system(pexprParserParser)
 *  - 
 void
      pexprParserParser->variableDeclaration(pexprParserParser)
 *  - 
 void
      pexprParserParser->arrayDeclaration(pexprParserParser)
 *  - 
 void
      pexprParserParser->transition(pexprParserParser)
 *  - 
 its::Sequence
      pexprParserParser->body(pexprParserParser)
 *  - 
 its::Ite
      pexprParserParser->iteAction(pexprParserParser)
 *  - 
 void
      pexprParserParser->transient(pexprParserParser)
 *  - 
 its::BoolExpression
      pexprParserParser->boolOr(pexprParserParser)
 *  - 
 its::BoolExpression
      pexprParserParser->boolAnd(pexprParserParser)
 *  - 
 its::BoolExpression
      pexprParserParser->boolNot(pexprParserParser)
 *  - 
 its::BoolExpression
      pexprParserParser->boolPrimary(pexprParserParser)
 *  - 
 its::BoolExpression
      pexprParserParser->comparison(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->bit_or(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->bitxor(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->bit_and(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->bitshift(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->addition(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->multiplication(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->unaryMinus(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->power(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->intPrimary(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->varAccess(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->variableRef(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->arrayVarAccess(pexprParserParser)
 *  - 
 its::IntExpression
      pexprParserParser->wrapBool(pexprParserParser)
 *  - 
 std::vector<int>
      pexprParserParser->initValues(pexprParserParser)
 *  - 
 its::BoolExprType
      pexprParserParser->comparisonOperators(pexprParserParser)
 *  - 
 exprParserParser_qualifiedName_return
      pexprParserParser->qualifiedName(pexprParserParser)
 *  - 
 int
      pexprParserParser->integer(pexprParserParser)
 * 
 * 
 *
 * The return type for any particular rule is of course determined by the source
 * grammar file.
 */
// [The "BSD license"]
// Copyright (c) 2005-2009 Jim Idle, Temporal Wave LLC
// http://www.temporal-wave.com
// http://www.linkedin.com/in/jimidle
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef	_exprParserParser_H
#define _exprParserParser_H
/* =============================================================================
 * Standard antlr3 C runtime definitions
 */
#include    <antlr3.h>

/* End of standard antlr 3 runtime definitions
 * =============================================================================
 */

#ifdef __cplusplus
extern "C" {
#endif

// Forward declare the context typedef so that we can use it before it is
// properly defined. Delegators and delegates (from import statements) are
// interdependent and their context structures contain pointers to each other
// C only allows such things to be declared if you pre-declare the typedef.
//
typedef struct exprParserParser_Ctx_struct exprParserParser, * pexprParserParser;




  #include "its/gal/GAL.hh"
  #include "its/composite/Composite.hh"
  #include "its/ITSModel.hh"
  #include <iostream>
  #include <cstdlib>


#ifdef	ANTLR3_WINDOWS
// Disable: Unreferenced parameter,							- Rules with parameters that are not used
//          constant conditional,							- ANTLR realizes that a prediction is always true (synpred usually)
//          initialized but unused variable					- tree rewrite variables declared but not needed
//          Unreferenced local variable						- lexer rule declares but does not always use _type
//          potentially unitialized variable used			- retval always returned from a rule
//			unreferenced local function has been removed	- susually getTokenNames or freeScope, they can go without warnigns
//
// These are only really displayed at warning level /W4 but that is the code ideal I am aiming at
// and the codegen must generate some of these warnings by necessity, apart from 4100, which is
// usually generated when a parser rule is given a parameter that it does not use. Mostly though
// this is a matter of orthogonality hence I disable that one.
//
#pragma warning( disable : 4100 )
#pragma warning( disable : 4101 )
#pragma warning( disable : 4127 )
#pragma warning( disable : 4189 )
#pragma warning( disable : 4505 )
#pragma warning( disable : 4701 )
#endif

/* ========================
 * BACKTRACKING IS ENABLED
 * ========================
 */

typedef struct exprParserParser_qualifiedName_return_struct
{
    /** Generic return elements for ANTLR3 rules that are not in tree parsers or returning trees
     */
    pANTLR3_COMMON_TOKEN    start;
    pANTLR3_COMMON_TOKEN    stop;
    std::string res;
}
    exprParserParser_qualifiedName_return;





/** Context tracking structure for 
exprParserParser

 */
struct exprParserParser_Ctx_struct
{
    /** Built in ANTLR3 context tracker contains all the generic elements
     *  required for context tracking.
     */
    pANTLR3_PARSER   pParser;

     void
     (*setGAL)	(struct exprParserParser_Ctx_struct * ctx, const its::GAL * g);

     void
     (*setModel)	(struct exprParserParser_Ctx_struct * ctx, const its::ITSModel * g);

     void
     (*specification)	(struct exprParserParser_Ctx_struct * ctx);

     its::Composite *
     (*composite)	(struct exprParserParser_Ctx_struct * ctx);

     its::GAL*
     (*system)	(struct exprParserParser_Ctx_struct * ctx);

     void
     (*variableDeclaration)	(struct exprParserParser_Ctx_struct * ctx);

     void
     (*arrayDeclaration)	(struct exprParserParser_Ctx_struct * ctx);

     void
     (*transition)	(struct exprParserParser_Ctx_struct * ctx);

     its::Sequence
     (*body)	(struct exprParserParser_Ctx_struct * ctx);

     its::Ite
     (*iteAction)	(struct exprParserParser_Ctx_struct * ctx);

     void
     (*transient)	(struct exprParserParser_Ctx_struct * ctx);

     its::BoolExpression
     (*boolOr)	(struct exprParserParser_Ctx_struct * ctx);

     its::BoolExpression
     (*boolAnd)	(struct exprParserParser_Ctx_struct * ctx);

     its::BoolExpression
     (*boolNot)	(struct exprParserParser_Ctx_struct * ctx);

     its::BoolExpression
     (*boolPrimary)	(struct exprParserParser_Ctx_struct * ctx);

     its::BoolExpression
     (*comparison)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*bit_or)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*bitxor)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*bit_and)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*bitshift)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*addition)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*multiplication)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*unaryMinus)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*power)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*intPrimary)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*varAccess)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*variableRef)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*arrayVarAccess)	(struct exprParserParser_Ctx_struct * ctx);

     its::IntExpression
     (*wrapBool)	(struct exprParserParser_Ctx_struct * ctx);

     std::vector<int>
     (*initValues)	(struct exprParserParser_Ctx_struct * ctx);

     its::BoolExprType
     (*comparisonOperators)	(struct exprParserParser_Ctx_struct * ctx);

     exprParserParser_qualifiedName_return
     (*qualifiedName)	(struct exprParserParser_Ctx_struct * ctx);

     int
     (*integer)	(struct exprParserParser_Ctx_struct * ctx);

     ANTLR3_BOOLEAN
     (*synpred1_exprParser)	(struct exprParserParser_Ctx_struct * ctx);

     ANTLR3_BOOLEAN
     (*synpred2_exprParser)	(struct exprParserParser_Ctx_struct * ctx);
    // Delegated rules

    const char * (*getGrammarFileName)();
    void            (*reset)  (struct exprParserParser_Ctx_struct * ctx);
    void	    (*free)   (struct exprParserParser_Ctx_struct * ctx);
};

// Function protoypes for the constructor functions that external translation units
// such as delegators and delegates may wish to call.
//
ANTLR3_API pexprParserParser exprParserParserNew         (
pANTLR3_COMMON_TOKEN_STREAM
 instream);
ANTLR3_API pexprParserParser exprParserParserNewSSD      (
pANTLR3_COMMON_TOKEN_STREAM
 instream, pANTLR3_RECOGNIZER_SHARED_STATE state);

/** Symbolic definitions of all the tokens that the 
parser
 will work with.
 * \{
 *
 * Antlr will define EOF, but we can't use that as it it is too common in
 * in C header files and that would be confusing. There is no way to filter this out at the moment
 * so we just undef it here for now. That isn't the value we get back from C recognizers
 * anyway. We are looking for ANTLR3_TOKEN_EOF.
 */
#ifdef	EOF
#undef	EOF
#endif
#ifdef	Tokens
#undef	Tokens
#endif
#define EOF      -1
#define T__10      10
#define T__11      11
#define T__12      12
#define T__13      13
#define T__14      14
#define T__15      15
#define T__16      16
#define T__17      17
#define T__18      18
#define T__19      19
#define T__20      20
#define T__21      21
#define T__22      22
#define T__23      23
#define T__24      24
#define T__25      25
#define T__26      26
#define T__27      27
#define T__28      28
#define T__29      29
#define T__30      30
#define T__31      31
#define T__32      32
#define T__33      33
#define T__34      34
#define T__35      35
#define T__36      36
#define T__37      37
#define T__38      38
#define T__39      39
#define T__40      40
#define T__41      41
#define T__42      42
#define T__43      43
#define T__44      44
#define T__45      45
#define T__46      46
#define T__47      47
#define T__48      48
#define T__49      49
#define T__50      50
#define T__51      51
#define T__52      52
#define T__53      53
#define T__54      54
#define T__55      55
#define T__56      56
#define T__57      57
#define T__58      58
#define T__59      59
#define ID      4
#define INT      5
#define ML_COMMENT      6
#define SL_COMMENT      7
#define STRING      8
#define WS      9
#ifdef	EOF
#undef	EOF
#define	EOF	ANTLR3_TOKEN_EOF
#endif

#ifndef TOKENSOURCE
#define TOKENSOURCE(lxr) lxr->pLexer->rec->state->tokSource
#endif

/* End of token definitions for exprParserParser
 * =============================================================================
 */
/** } */

#ifdef __cplusplus
}
#endif

#endif

/* END - Note:Keep extra line feed to satisfy UNIX systems */