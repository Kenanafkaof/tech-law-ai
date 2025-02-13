import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface RejectionChartProps {
  data: {
    rejection_analysis: {
      counts: {
        [key: string]: number
      }
    }
  }
}

const rejectionDescriptions = {
  '101': 'Subject Matter Eligibility - Is the invention eligible for patent protection?',
  '102': 'Novelty - Is the invention new and not previously disclosed?',
  '103': 'Obviousness - Would the invention be obvious to someone skilled in the field?',
  '112': 'Written Description/Enablement - Is the invention clearly described?'
}

export function RejectionChart({ data }: RejectionChartProps) {
  const chartData = Object.entries(data.rejection_analysis.counts).map(([type, count]) => ({
    type: `§${type}`,
    count,
    description: rejectionDescriptions[type as keyof typeof rejectionDescriptions]
  }))

  return (
    <Card className="col-span-3">
      <CardHeader>
        <CardTitle>Rejection Analysis</CardTitle>
        <CardDescription>
          Distribution of rejection types across patent applications
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="type" />
              <YAxis />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload
                    return (
                      <div className="rounded-lg border bg-background p-2 shadow-sm">
                        <div className="grid grid-cols-2 gap-2">
                          <div className="flex flex-col">
                            <span className="text-[0.70rem] uppercase text-muted-foreground">
                              {data.type}
                            </span>
                            <span className="font-bold text-muted-foreground">
                              {data.count.toLocaleString()} rejections
                            </span>
                          </div>
                        </div>
                        <div className="mt-2 text-xs text-muted-foreground">
                          {data.description}
                        </div>
                      </div>
                    )
                  }
                  return null
                }}
              />
              <Bar
                dataKey="count"
                fill="#2563eb"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}