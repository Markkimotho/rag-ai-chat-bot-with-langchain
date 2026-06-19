import { useReducer, useState } from "react";
import { api, ApiError } from "../../api/client";
import { useModelContext } from "../../context/ModelContext";
import { FeedbackCard } from "./FeedbackCard";
import { QuestionCard } from "./QuestionCard";
import { ResultsCard } from "./ResultsCard";
import { SetupPanel, type SetupValues } from "./SetupPanel";
import {
  initialState,
  quizReducer,
  scoreSummary,
  type QuizConfig,
} from "./quizReducer";
import sectionStyles from "../Section.module.css";
import styles from "./QuizPrep.module.css";

export function QuizPrep() {
  const { selected } = useModelContext();
  const [state, dispatch] = useReducer(quizReducer, initialState);
  const [generating, setGenerating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = async (v: SetupValues) => {
    setGenerating(true);
    setError(null);
    try {
      const questions = await api.quizGenerate({
        topic: v.topic,
        question_type: v.questionType,
        n: v.n,
        difficulty: v.difficulty,
        exam_type: v.examType,
        model: selected || undefined,
      });
      if (!questions.length) {
        setError(
          "No questions generated. Make sure your material covers this topic.",
        );
        return;
      }
      const config: QuizConfig = {
        topic: v.topic,
        questionType: v.questionType,
        difficulty: v.difficulty,
        examType: v.examType,
      };
      dispatch({ type: "START", questions, config });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Generation failed.");
    } finally {
      setGenerating(false);
    }
  };

  const submitAnswer = async (studentAnswer: string) => {
    const q = state.questions[state.idx];
    const correct = q.correct ?? q.sample_answer ?? "";
    const qType =
      q.type === "mixed" ? "short_answer" : q.type; // questions are concrete
    setSubmitting(true);
    setError(null);
    try {
      const result = await api.quizValidate({
        question: q.question,
        correct_answer: correct,
        student_answer: studentAnswer,
        question_type: qType,
        model: selected || undefined,
      });
      dispatch({
        type: "SUBMIT",
        record: {
          questionIdx: state.idx,
          question: q.question,
          questionType: qType,
          studentAnswer,
          correctAnswer: String(correct),
          result,
        },
      });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not grade answer.");
    } finally {
      setSubmitting(false);
    }
  };

  const correctSoFar = scoreSummary(state.answers).correct;
  const current = state.questions[state.idx];
  const isLast = state.idx + 1 >= state.questions.length;

  return (
    <div className={sectionStyles.section}>
      <div className={styles.scroll}>
        <div className={styles.container}>
          {state.phase === "setup" && (
            <SetupPanel
              generating={generating}
              error={error}
              onGenerate={generate}
            />
          )}

          {state.phase === "question" && current && (
            <>
              <QuestionCard
                question={current}
                idx={state.idx}
                total={state.questions.length}
                correctSoFar={correctSoFar}
                submitting={submitting}
                onSubmit={submitAnswer}
              />
              {error && <div className={styles.error}>{error}</div>}
            </>
          )}

          {state.phase === "feedback" && state.lastResult && (
            <FeedbackCard
              result={state.lastResult}
              question={state.questions[state.idx]}
              isLast={isLast}
              onNext={() => dispatch({ type: "NEXT" })}
            />
          )}

          {state.phase === "results" && (
            <ResultsCard
              answers={state.answers}
              topic={state.config?.topic ?? ""}
              onRetake={() => dispatch({ type: "RETAKE" })}
              onNew={() => dispatch({ type: "RESET" })}
            />
          )}
        </div>
      </div>
    </div>
  );
}
