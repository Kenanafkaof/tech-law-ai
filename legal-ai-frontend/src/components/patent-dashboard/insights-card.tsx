import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Info } from "lucide-react"

interface InsightsCardProps {
  data: {
    rejection_analysis: {
      most_common: [string, number]
    }
    temporal_analysis: {
      yearly_trends: {
        [key: string]: number
      }
    }
    claim_analysis: {
      most_rejected_claims: [number, number][]
    }
  }
}

const rejectionAdvice = {
  '101': 'Focus on technical improvements and practical applications. Avoid claiming abstract ideas without significant additional elements.',
  '102': 'Ensure your invention has unique features not found in existing patents or publications.',
  '103': 'Emphasize the non-obvious combinations and unexpected results of your invention.',
  '112': 'Provide clear, detailed descriptions and examples of how to implement your invention.'
}

export function InsightsCard({ data }: InsightsCardProps) {
  const currentYear = '2024'
  const previousYear = '2023'
  const yearlyTrends = data.temporal_analysis.yearly_trends
  const trendDirection = yearlyTrends[currentYear] > yearlyTrends[previousYear] ? 'increased' : 'decreased'
  
  const topClaims = data.claim_analysis.most_rejected_claims
    .slice(0, 3)
    .map(([claim]) => claim)
    .join(', ')

  return (
    <Card className="col-span-3">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Info className="h-4 w-4" />
          Key Insights & Recommendations
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-lg bg-blue-50 p-4 dark:bg-blue-950">
          <h3 className="font-semibold mb-2">
            Most Common Rejection: §{data.rejection_analysis.most_common[0]}
          </h3>
          <p className="text-sm text-muted-foreground">
            {rejectionAdvice[data.rejection_analysis.most_common[0] as keyof typeof rejectionAdvice]}
          </p>
        </div>
        
        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-lg bg-green-50 p-4 dark:bg-green-950">
            <h3 className="font-semibold mb-2">Current Trends</h3>
            <p className="text-sm text-muted-foreground">
              Applications have {trendDirection} compared to last year 
              ({yearlyTrends[currentYear].toLocaleString()} vs {yearlyTrends[previousYear].toLocaleString()} applications).
            </p>
          </div>
          
          <div className="rounded-lg bg-purple-50 p-4 dark:bg-purple-950">
            <h3 className="font-semibold mb-2">Claims Strategy</h3>
            <p className="text-sm text-muted-foreground">
              Pay special attention to claims {topClaims} as they face the most scrutiny during examination.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

